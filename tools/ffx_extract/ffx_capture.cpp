// Records what AMD FSR Frame Generation 4 (amd_fidelityfx_framegeneration_vk.dll) does on the GPU, headless.
//
//   ffx_capture DLL OUT_DIR [--frames N] [--size WxH] [--spoof-amd] [--validate] [--inputs DIR] [--dump-passes F A B] [--dump-images F] [--patch-cb]
//
// The DLL is loaded directly (it exports the FFX API). Its vulkan-1.dll import table is patched and the
// vkGetDeviceProcAddr it is given is wrapped, so every Vulkan call it makes goes through the hooks below. Written to
// OUT_DIR:
//   modules/<hash>.spv      every shader module it creates
//   pipelines.txt           pipeline id -> module hash, entry point, specialisation data
//   log.txt                 every dispatch / copy / fill / update in submission order, with resolved descriptors,
//                           push constants and the bytes of small host-visible buffers it reads (constant buffers)
//   uploads/<n>.bin         the source bytes of every buffer-to-buffer / buffer-to-image copy from host memory
//   buffers/<id>.bin        the final contents of every buffer of 64 KiB or more (after the last frame)
//   output_<frame>.rgba     the generated frame (RGBA8)
// Inputs are synthetic (a textured frame moving 8 px right per frame) unless --inputs DIR holds
// color_<f>.rgba / depth_<f>.f32 / mv_<f>.rg32f at the chosen size.
//
// The FidelityFX SDK v2.4.0 licence lists these DLLs and the API headers under MIT (docs/FFX_MLFG.md).

#include <vulkan/vulkan.h>
#include <windows.h>

#include "api/include/ffx_api.h"
#include "api/include/ffx_api_loader.h"
#include "api/include/vk/ffx_api_vk.h"
#include "framegeneration/include/ffx_framegeneration.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#define CHECK_VK(x)                                                                       \
    do {                                                                                  \
        VkResult r_ = (x);                                                                \
        if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(2); } \
    } while (0)

static std::string gOut;
static bool        gSpoof = false;
static bool        gPatchCb = false;
static int         gFrame = 0;
static std::recursive_mutex gLock;
static FILE*       gLog;

// ---------------------------------------------------------------- tracked state
struct BufferInfo { uint32_t id; VkDeviceSize size; VkBufferUsageFlags usage; VkDeviceMemory mem = 0; VkDeviceSize memOffset = 0; };
struct ImageInfo  { uint32_t id; VkImageCreateInfo ci; };
struct ViewInfo   { uint32_t id; VkImage image; VkFormat format; };
struct Desc       { VkDescriptorType type; VkBuffer buffer; VkDeviceSize offset, range; VkImageView view; VkSampler sampler; };
struct Pipeline   { uint32_t id; std::string module, entry; std::vector<uint8_t> spec; VkPipelineLayout layout; };
struct CbPatch    { VkBuffer buf; VkDeviceSize off; int kind; };  // --patch-cb: kind 1 = CsDilate, 2 = CsFrameInterpolationConstants
struct Record     { std::string text; std::vector<std::pair<VkBuffer, std::pair<VkDeviceSize, VkDeviceSize>>> snap; std::vector<std::pair<VkBuffer, std::pair<VkDeviceSize, VkDeviceSize>>> upload; std::vector<CbPatch> patch; };
struct CbState
{
    VkPipeline pipeline = 0;
    std::map<uint32_t, std::pair<VkDescriptorSet, uint32_t>> sets; // set index -> (set, first dynamic offset index)
    std::vector<uint32_t> dynOffsets;
    std::map<uint32_t, std::vector<uint32_t>> setDyn;               // set index -> its dynamic offsets
    std::vector<uint8_t> push;
    std::map<uint32_t, std::map<uint64_t, Desc>> pushDesc;          // push descriptors per set
    std::vector<Record> records;
};

static std::unordered_map<VkBuffer, BufferInfo>                       gBuffers;
static std::unordered_map<VkImage, ImageInfo>                         gImages;
static std::unordered_map<VkImageView, ViewInfo>                      gViews;
static std::unordered_map<VkDeviceMemory, void*>                      gMapped;
static std::unordered_map<VkShaderModule, std::string>                gModules;
static std::unordered_map<VkPipeline, Pipeline>                       gPipelines;
static std::unordered_map<VkDescriptorSet, std::map<uint64_t, Desc>>  gSets;
static std::unordered_map<VkCommandBuffer, CbState>                   gCbs;
static uint32_t gNextBuffer = 0, gNextImage = 0, gNextView = 0, gNextPipeline = 0, gSubmits = 0, gUploads = 0;
static uint64_t gSeq = 0;

// real entry points
static PFN_vkGetDeviceProcAddr gRealGDPA;
static PFN_vkGetInstanceProcAddr gRealGIPA;
static VkInstance gInstance;
#define REAL(name) static PFN_##name r_##name
REAL(vkCreateShaderModule); REAL(vkCreateComputePipelines); REAL(vkCmdBindPipeline); REAL(vkCmdBindDescriptorSets);
REAL(vkUpdateDescriptorSets); REAL(vkCmdPushConstants); REAL(vkCmdDispatch); REAL(vkCmdDispatchIndirect);
REAL(vkCmdCopyBuffer); REAL(vkCmdCopyBufferToImage); REAL(vkCmdUpdateBuffer); REAL(vkCmdFillBuffer);
REAL(vkCreateBuffer); REAL(vkBindBufferMemory); REAL(vkMapMemory); REAL(vkCreateImage); REAL(vkCreateImageView);
REAL(vkQueueSubmit); REAL(vkQueueSubmit2); REAL(vkBeginCommandBuffer); REAL(vkGetPhysicalDeviceProperties);
REAL(vkGetPhysicalDeviceProperties2); REAL(vkCmdPushDescriptorSetKHR); REAL(vkCmdClearColorImage); REAL(vkCmdCopyImage);

static std::string Hex(const void* p, size_t n)
{
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { uint8_t b = ((const uint8_t*)p)[i]; s += d[b >> 4]; s += d[b & 15]; }
    return s;
}

static uint64_t Fnv(const void* p, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
}

static const uint8_t* HostPtr(VkBuffer b, VkDeviceSize off)
{
    auto it = gBuffers.find(b);
    if (it == gBuffers.end() || !it->second.mem) return nullptr;
    auto m = gMapped.find(it->second.mem);
    return m == gMapped.end() ? nullptr : (const uint8_t*)m->second + it->second.memOffset + off;
}

static std::string DescText(const Desc& d, std::vector<std::pair<VkBuffer, std::pair<VkDeviceSize, VkDeviceSize>>>* snap, uint32_t dyn)
{
    char t[160];
    if (d.buffer)
    {
        auto& bi = gBuffers[d.buffer];
        VkDeviceSize off = d.offset + dyn, range = d.range == VK_WHOLE_SIZE ? bi.size - off : d.range;
        snprintf(t, sizeof(t), "t%d:buf%u+%llu,%llu", int(d.type), bi.id, (unsigned long long)off, (unsigned long long)range);
        if (snap && range <= 65536 && HostPtr(d.buffer, 0)) snap->push_back({d.buffer, {off, range}});
    }
    else if (d.view)
    {
        auto& v = gViews[d.view];
        snprintf(t, sizeof(t), "t%d:img%u/v%u/f%d", int(d.type), gImages[v.image].id, v.id, int(v.format));
    }
    else snprintf(t, sizeof(t), "t%d:sampler", int(d.type));
    return t;
}

// ---------------------------------------------------------------- hooks
static VKAPI_ATTR VkResult VKAPI_CALL H_vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* m)
{
    VkResult r = r_vkCreateShaderModule(d, ci, a, m);
    if (r == VK_SUCCESS)
    {
        char h[17]; snprintf(h, sizeof(h), "%016llx", (unsigned long long)Fnv(ci->pCode, ci->codeSize));
        std::lock_guard<std::recursive_mutex> l(gLock);
        gModules[*m] = h;
        fs::path p = fs::path(gOut) / "modules" / (std::string(h) + ".spv");
        if (!fs::exists(p)) { FILE* f = fopen(p.string().c_str(), "wb"); fwrite(ci->pCode, 1, ci->codeSize, f); fclose(f); }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkCreateComputePipelines(VkDevice d, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* p)
{
    VkResult r = r_vkCreateComputePipelines(d, c, n, ci, a, p);
    std::lock_guard<std::recursive_mutex> l(gLock);
    FILE* f = fopen((fs::path(gOut) / "pipelines.txt").string().c_str(), "a");
    for (uint32_t i = 0; i < n; ++i)
    {
        if (!p[i]) continue;
        Pipeline pl{gNextPipeline++, gModules[ci[i].stage.module], ci[i].stage.pName, {}, ci[i].layout};
        if (auto s = ci[i].stage.pSpecializationInfo)
        {
            pl.spec.assign((const uint8_t*)s->pData, (const uint8_t*)s->pData + s->dataSize);
            std::string m;
            for (uint32_t k = 0; k < s->mapEntryCount; ++k)
                m += " " + std::to_string(s->pMapEntries[k].constantID) + "@" + std::to_string(s->pMapEntries[k].offset);
            fprintf(f, "pipe%u %s %s spec%s =%s\n", pl.id, pl.module.c_str(), pl.entry.c_str(), m.c_str(), Hex(s->pData, s->dataSize).c_str());
        }
        else fprintf(f, "pipe%u %s %s\n", pl.id, pl.module.c_str(), pl.entry.c_str());
        gPipelines[p[i]] = pl;
    }
    fclose(f);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p)
{
    r_vkCmdBindPipeline(cb, bp, p);
    std::lock_guard<std::recursive_mutex> l(gLock);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) gCbs[cb].pipeline = p;
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout lay, uint32_t first, uint32_t n, const VkDescriptorSet* s, uint32_t nd, const uint32_t* dyn)
{
    r_vkCmdBindDescriptorSets(cb, bp, lay, first, n, s, nd, dyn);
    if (bp != VK_PIPELINE_BIND_POINT_COMPUTE) return;
    std::lock_guard<std::recursive_mutex> l(gLock);
    auto& st = gCbs[cb];
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        st.sets[first + i] = {s[i], 0};
        std::vector<uint32_t> mine;
        // dynamic offsets are consumed in binding order of the set's dynamic descriptors
        for (auto& [key, d] : gSets[s[i]])
            if ((d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) && k < nd)
                mine.push_back(dyn[k++]);
        st.setDyn[first + i] = mine;
    }
}

static VKAPI_ATTR void VKAPI_CALL H_vkUpdateDescriptorSets(VkDevice d, uint32_t nw, const VkWriteDescriptorSet* w, uint32_t nc, const VkCopyDescriptorSet* c)
{
    r_vkUpdateDescriptorSets(d, nw, w, nc, c);
    std::lock_guard<std::recursive_mutex> l(gLock);
    for (uint32_t i = 0; i < nw; ++i)
        for (uint32_t k = 0; k < w[i].descriptorCount; ++k)
        {
            Desc e{w[i].descriptorType, 0, 0, 0, 0, 0};
            if (w[i].pBufferInfo) e.buffer = w[i].pBufferInfo[k].buffer, e.offset = w[i].pBufferInfo[k].offset, e.range = w[i].pBufferInfo[k].range;
            else if (w[i].pImageInfo) e.view = w[i].pImageInfo[k].imageView, e.sampler = w[i].pImageInfo[k].sampler;
            gSets[w[i].dstSet][(uint64_t(w[i].dstBinding) << 32) | (w[i].dstArrayElement + k)] = e;
        }
    for (uint32_t i = 0; i < nc; ++i)
        for (uint32_t k = 0; k < c[i].descriptorCount; ++k)
            gSets[c[i].dstSet][(uint64_t(c[i].dstBinding) << 32) | (c[i].dstArrayElement + k)] =
                gSets[c[i].srcSet][(uint64_t(c[i].srcBinding) << 32) | (c[i].srcArrayElement + k)];
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout lay, uint32_t set, uint32_t nw, const VkWriteDescriptorSet* w)
{
    r_vkCmdPushDescriptorSetKHR(cb, bp, lay, set, nw, w);
    std::lock_guard<std::recursive_mutex> l(gLock);
    auto& m = gCbs[cb].pushDesc[set];
    for (uint32_t i = 0; i < nw; ++i)
        for (uint32_t k = 0; k < w[i].descriptorCount; ++k)
        {
            Desc e{w[i].descriptorType, 0, 0, 0, 0, 0};
            if (w[i].pBufferInfo) e.buffer = w[i].pBufferInfo[k].buffer, e.offset = w[i].pBufferInfo[k].offset, e.range = w[i].pBufferInfo[k].range;
            else if (w[i].pImageInfo) e.view = w[i].pImageInfo[k].imageView, e.sampler = w[i].pImageInfo[k].sampler;
            m[(uint64_t(w[i].dstBinding) << 32) | (w[i].dstArrayElement + k)] = e;
        }
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout lay, VkShaderStageFlags s, uint32_t off, uint32_t size, const void* v)
{
    r_vkCmdPushConstants(cb, lay, s, off, size, v);
    std::lock_guard<std::recursive_mutex> l(gLock);
    auto& p = gCbs[cb].push;
    if (p.size() < off + size) p.resize(off + size);
    memcpy(p.data() + off, v, size);
}

static void RecordDispatch(VkCommandBuffer cb, const char* what)
{
    std::lock_guard<std::recursive_mutex> l(gLock);
    auto& st = gCbs[cb];
    Record r;
    auto& pl = gPipelines[st.pipeline];
    r.text = std::string(what) + " pipe" + std::to_string(pl.id) + " " + pl.entry;
    if (!st.push.empty()) r.text += " pc=" + Hex(st.push.data(), st.push.size());
    for (auto& [si, sp] : st.sets)
    {
        size_t k = 0;
        auto& dyn = st.setDyn[si];
        for (auto& [key, d] : gSets[sp.first])
        {
            uint32_t dv = 0;
            if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) dv = k < dyn.size() ? dyn[k++] : 0;
            r.text += " s" + std::to_string(si) + "b" + std::to_string(key >> 32) + "[" + std::to_string(uint32_t(key)) + "]=" + DescText(d, &r.snap, dv);
            if (gPatchCb && (key >> 32) == 256 && d.buffer && d.range == 80 && HostPtr(d.buffer, 0))
            {
                int kind = pl.entry == "main" ? 1 : pl.entry.rfind("mlfi_", 0) == 0 ? 2 : 0;
                if (kind == 2 && gFrame == 0) kind = 0;  // the first frame really is a reset
                if (kind) r.patch.push_back({d.buffer, d.offset + dv, kind});
            }
        }
    }
    for (auto& [si, m] : st.pushDesc)
        for (auto& [key, d] : m)
            r.text += " p" + std::to_string(si) + "b" + std::to_string(key >> 32) + "[" + std::to_string(uint32_t(key)) + "]=" + DescText(d, &r.snap, 0);
    st.records.push_back(std::move(r));
}

// --dump-passes FRAME FIRST LAST: after each ML pass FIRST..LAST of frame FRAME, copy the whole scratch buffer
// (binding 128) to a host-visible buffer; written as passes/<frame>_<pass>.bin when the frame is done.
static int gDumpFrame = -1, gDumpFirst = 0, gDumpLast = -1;
static VkDevice gDev;
static VkPhysicalDeviceMemoryProperties gMemProps;
struct PassDump { int pass; VkBuffer buf; VkDeviceMemory mem; void* ptr; VkDeviceSize size; };
static std::vector<PassDump> gPassDumps;

static VkBuffer ScratchOf(CbState& st)
{
    for (auto& [si, sp] : st.sets)
        for (auto& [key, d] : gSets[sp.first])
            if ((key >> 32) == 128 && d.buffer) return d.buffer;
    return 0;
}

static void DumpAfter(VkCommandBuffer cb, int pass)
{
    auto& st = gCbs[cb];
    VkBuffer scratch = ScratchOf(st);
    if (!scratch) return;
    VkDeviceSize size = gBuffers[scratch].size;
    PassDump d{pass, 0, 0, nullptr, size};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    auto createBuffer = (PFN_vkCreateBuffer)gRealGDPA(gDev, "vkCreateBuffer");
    CHECK_VK(createBuffer(gDev, &bi, nullptr, &d.buf));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(gDev, d.buf, &r);
    uint32_t type = 0;
    for (uint32_t i = 0; i < gMemProps.memoryTypeCount; ++i)
        if ((r.memoryTypeBits & (1u << i)) && (gMemProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) ==
                                                  (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) { type = i; break; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, type};
    CHECK_VK(vkAllocateMemory(gDev, &ai, nullptr, &d.mem));
    CHECK_VK(vkBindBufferMemory(gDev, d.buf, d.mem, 0));
    CHECK_VK(vkMapMemory(gDev, d.mem, 0, size, 0, &d.ptr));
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy c{0, 0, size};
    r_vkCmdCopyBuffer(cb, scratch, d.buf, 1, &c);
    VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);
    gPassDumps.push_back(d);
}

static void WritePassDumps()
{
    fs::create_directories(fs::path(gOut) / "passes");
    for (auto& d : gPassDumps)
    {
        char n[48]; snprintf(n, sizeof(n), "%d_%03d.bin", gDumpFrame, d.pass);
        FILE* f = fopen((fs::path(gOut) / "passes" / n).string().c_str(), "wb");
        fwrite(d.ptr, 1, size_t(d.size), f); fclose(f);
        vkDestroyBuffer(gDev, d.buf, nullptr); vkFreeMemory(gDev, d.mem, nullptr);
    }
    gPassDumps.clear();
}

// --dump-images FRAME: around each dispatch of frame FRAME whose entry is "main" or mlfi_*_pass151/152/190, copy every
// image it binds: inputs (binding < 128) before the dispatch, storage images bound for write (binding >= 128) after it.
// Written as images/<frame>_<seq>_<entry>_b<binding>.raw (tightly packed mip 0 / layer 0) with images/index.txt.
static int gImgFrame = -1, gImgSeq = 0;
struct ImgDump { std::string name, line; VkBuffer buf; VkDeviceMemory mem; void* ptr; VkDeviceSize size; };
static std::vector<ImgDump> gImgDumps;

static uint32_t TexelBytes(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_D32_SFLOAT: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R64_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 0;
    }
}

static bool WantImages(const std::string& e)
{
    if (e == "main") return true;
    if (e.rfind("mlfi_", 0) != 0) return false;
    for (const char* s : {"_pass151", "_pass152", "_pass190"})
        if (e.size() >= strlen(s) && e.compare(e.size() - strlen(s), strlen(s), s) == 0) return true;
    return false;
}

static void DumpImages(VkCommandBuffer cb, bool outputs)
{
    auto& st = gCbs[cb];
    auto& pl = gPipelines[st.pipeline];
    std::vector<std::pair<uint32_t, Desc>> sel;
    auto take = [&](uint64_t key, const Desc& d) {
        uint32_t b = uint32_t(key >> 32);
        if (!d.view || uint32_t(key) != 0) return;
        if (outputs ? (b >= 128 && d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) : (b < 128)) sel.push_back({b, d});
    };
    for (auto& [si, sp] : st.sets) for (auto& [key, d] : gSets[sp.first]) take(key, d);
    for (auto& [si, m] : st.pushDesc) for (auto& [key, d] : m) take(key, d);
    if (sel.empty()) return;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    for (auto& [b, d] : sel)
    {
        auto& v = gViews[d.view];
        auto it = gImages.find(v.image);
        if (it == gImages.end()) continue;
        auto& ci = it->second.ci;
        uint32_t tb = TexelBytes(ci.format);
        if (!tb || ci.imageType != VK_IMAGE_TYPE_2D) continue;
        VkDeviceSize size = VkDeviceSize(ci.extent.width) * ci.extent.height * tb;
        ImgDump o{};
        o.size = size;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        auto createBuffer = (PFN_vkCreateBuffer)gRealGDPA(gDev, "vkCreateBuffer");
        CHECK_VK(createBuffer(gDev, &bi, nullptr, &o.buf));
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(gDev, o.buf, &r);
        uint32_t type = 0;
        for (uint32_t i = 0; i < gMemProps.memoryTypeCount; ++i)
            if ((r.memoryTypeBits & (1u << i)) && (gMemProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) ==
                                                      (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) { type = i; break; }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, type};
        CHECK_VK(vkAllocateMemory(gDev, &ai, nullptr, &o.mem));
        CHECK_VK(vkBindBufferMemory(gDev, o.buf, o.mem, 0));
        CHECK_VK(vkMapMemory(gDev, o.mem, 0, size, 0, &o.ptr));
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {ci.extent.width, ci.extent.height, 1};
        vkCmdCopyImageToBuffer(cb, v.image, VK_IMAGE_LAYOUT_GENERAL, o.buf, 1, &c);
        char n[160]; snprintf(n, sizeof(n), "%d_%03d_%s_b%u.raw", gImgFrame, gImgSeq, pl.entry.c_str(), b);
        o.name = n;
        char t[320]; snprintf(t, sizeof(t), "%s %ux%u fmt %d viewfmt %d img%u v%u pipe%u %s t%d", n, ci.extent.width, ci.extent.height, int(ci.format),
                              int(v.format), it->second.id, v.id, pl.id, outputs ? "out" : "in", int(d.type));
        o.line = t;
        gImgDumps.push_back(o);
    }
    VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);
}

static void WriteImageDumps()
{
    fs::create_directories(fs::path(gOut) / "images");
    FILE* idx = fopen((fs::path(gOut) / "images" / "index.txt").string().c_str(), "a");
    for (auto& d : gImgDumps)
    {
        FILE* f = fopen((fs::path(gOut) / "images" / d.name).string().c_str(), "wb");
        fwrite(d.ptr, 1, size_t(d.size), f); fclose(f);
        fprintf(idx, "%s\n", d.line.c_str());
        vkDestroyBuffer(gDev, d.buf, nullptr); vkFreeMemory(gDev, d.mem, nullptr);
    }
    fclose(idx);
    gImgDumps.clear();
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
    char t[64]; snprintf(t, sizeof(t), "D %u,%u,%u", x, y, z);
    RecordDispatch(cb, t);
    bool imgs = false;
    if (gFrame == gImgFrame)
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        imgs = WantImages(gPipelines[gCbs[cb].pipeline].entry);
        if (imgs) DumpImages(cb, false);
    }
    r_vkCmdDispatch(cb, x, y, z);
    if (imgs)
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        DumpImages(cb, true);
        ++gImgSeq;
    }
    if (gFrame == gDumpFrame)
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        auto& e = gPipelines[gCbs[cb].pipeline].entry;
        auto p = e.find("_pass");
        if (e.rfind("mlfi_", 0) == 0 && p != std::string::npos)
        {
            int pass = atoi(e.c_str() + p + 5);
            if (pass >= gDumpFirst && pass <= gDumpLast) DumpAfter(cb, pass);
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off)
{
    char t[64]; snprintf(t, sizeof(t), "DI buf%u+%llu", gBuffers[b].id, (unsigned long long)off);
    RecordDispatch(cb, t);
    r_vkCmdDispatchIndirect(cb, b, off);
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* rg)
{
    r_vkCmdCopyBuffer(cb, s, d, n, rg);
    std::lock_guard<std::recursive_mutex> l(gLock);
    for (uint32_t i = 0; i < n; ++i)
    {
        Record r;
        char t[160];
        snprintf(t, sizeof(t), "C buf%u+%llu -> buf%u+%llu size %llu", gBuffers[s].id, (unsigned long long)rg[i].srcOffset, gBuffers[d].id,
                 (unsigned long long)rg[i].dstOffset, (unsigned long long)rg[i].size);
        r.text = t;
        if (HostPtr(s, 0)) r.upload.push_back({s, {rg[i].srcOffset, rg[i].size}});
        gCbs[cb].records.push_back(std::move(r));
    }
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout lay, uint32_t n, const VkBufferImageCopy* rg)
{
    r_vkCmdCopyBufferToImage(cb, s, d, lay, n, rg);
    std::lock_guard<std::recursive_mutex> l(gLock);
    for (uint32_t i = 0; i < n; ++i)
    {
        Record r;
        auto& ii = gImages[d];
        char t[200];
        snprintf(t, sizeof(t), "CI buf%u+%llu -> img%u mip%u %ux%ux%u fmt %d", gBuffers[s].id, (unsigned long long)rg[i].bufferOffset, ii.id,
                 rg[i].imageSubresource.mipLevel, rg[i].imageExtent.width, rg[i].imageExtent.height, rg[i].imageExtent.depth, int(ii.ci.format));
        r.text = t;
        if (HostPtr(s, 0))
        {
            VkDeviceSize sz = gBuffers[s].size - rg[i].bufferOffset; // whole tail; trimmed on disk below to the copy's needs
            r.upload.push_back({s, {rg[i].bufferOffset, sz}});
        }
        gCbs[cb].records.push_back(std::move(r));
    }
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer d, VkDeviceSize off, VkDeviceSize size, const void* data)
{
    r_vkCmdUpdateBuffer(cb, d, off, size, data);
    std::lock_guard<std::recursive_mutex> l(gLock);
    Record r;
    r.text = "UB buf" + std::to_string(gBuffers[d].id) + "+" + std::to_string(off) + " =" + Hex(data, size_t(size));
    gCbs[cb].records.push_back(std::move(r));
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer d, VkDeviceSize off, VkDeviceSize size, uint32_t v)
{
    r_vkCmdFillBuffer(cb, d, off, size, v);
    std::lock_guard<std::recursive_mutex> l(gLock);
    char t[120];
    snprintf(t, sizeof(t), "F buf%u+%llu size %lld = %08x", gBuffers[d].id, (unsigned long long)off, (long long)(size == VK_WHOLE_SIZE ? -1 : (long long)size), v);
    Record r; r.text = t;
    gCbs[cb].records.push_back(std::move(r));
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout lay, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r)
{
    r_vkCmdClearColorImage(cb, i, lay, c, n, r);
    std::lock_guard<std::recursive_mutex> l(gLock);
    Record rec; rec.text = "CLR img" + std::to_string(gImages[i].id) + " =" + Hex(c, 16);
    gCbs[cb].records.push_back(std::move(rec));
}

static VKAPI_ATTR void VKAPI_CALL H_vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy* rg)
{
    r_vkCmdCopyImage(cb, s, sl, d, dl, n, rg);
    std::lock_guard<std::recursive_mutex> l(gLock);
    Record rec; rec.text = "CPI img" + std::to_string(gImages[s].id) + " -> img" + std::to_string(gImages[d].id);
    gCbs[cb].records.push_back(std::move(rec));
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkCreateBuffer(VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* b)
{
    VkBufferCreateInfo c = *ci;
    c.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkResult r = r_vkCreateBuffer(d, &c, a, b);
    std::lock_guard<std::recursive_mutex> l(gLock);
    if (r == VK_SUCCESS) gBuffers[*b] = {gNextBuffer++, ci->size, ci->usage};
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkBindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize off)
{
    std::lock_guard<std::recursive_mutex> l(gLock);
    gBuffers[b].mem = m, gBuffers[b].memOffset = off;
    return r_vkBindBufferMemory(d, b, m, off);
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkMapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize off, VkDeviceSize size, VkMemoryMapFlags f, void** p)
{
    VkResult r = r_vkMapMemory(d, m, off, size, f, p);
    std::lock_guard<std::recursive_mutex> l(gLock);
    if (r == VK_SUCCESS) gMapped[m] = (uint8_t*)*p - off;
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkCreateImage(VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* i)
{
    VkResult r = r_vkCreateImage(d, ci, a, i);
    std::lock_guard<std::recursive_mutex> l(gLock);
    if (r == VK_SUCCESS)
    {
        gImages[*i] = {gNextImage++, *ci};
        fprintf(gLog, "IMG img%u %ux%ux%u mips %u layers %u fmt %d usage %x\n", gImages[*i].id, ci->extent.width, ci->extent.height,
                ci->extent.depth, ci->mipLevels, ci->arrayLayers, int(ci->format), ci->usage);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkCreateImageView(VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* v)
{
    VkResult r = r_vkCreateImageView(d, ci, a, v);
    std::lock_guard<std::recursive_mutex> l(gLock);
    if (r == VK_SUCCESS) gViews[*v] = {gNextView++, ci->image, ci->format};
    return r;
}

static void Flush(VkCommandBuffer cb)
{
    auto it = gCbs.find(cb);
    if (it == gCbs.end()) return;
    for (auto& r : it->second.records)
    {
        for (auto& p : r.patch)
        {
            // Without a swapchain ffxConfigure fails, so the DLL never latches the configure state: CsDilate.colorRect
            // stays (0,0)-(0,0) and every frame is flagged reset. Patch in what a configured full-screen context gives.
            uint8_t* q = (uint8_t*)HostPtr(p.buf, p.off);
            if (p.kind == 1) { float* rc = (float*)(q + 64); if (rc[0] == 0 && rc[1] == 0 && rc[2] == 0 && rc[3] == 0) rc[2] = rc[3] = 1.0f; }
            if (p.kind == 2) *(int32_t*)(q + 16) = 0;
        }
        fprintf(gLog, "%llu s%u %s", (unsigned long long)gSeq++, gSubmits, r.text.c_str());
        for (auto& [b, rg] : r.snap)
            if (auto p = HostPtr(b, rg.first)) fprintf(gLog, " | buf%u+%llu=%s", gBuffers[b].id, (unsigned long long)rg.first, Hex(p, size_t(rg.second)).c_str());
        for (auto& [b, rg] : r.upload)
            if (auto p = HostPtr(b, rg.first))
            {
                char n[32]; snprintf(n, sizeof(n), "%05u.bin", gUploads++);
                FILE* f = fopen((fs::path(gOut) / "uploads" / n).string().c_str(), "wb");
                fwrite(p, 1, size_t(rg.second), f); fclose(f);
                fprintf(gLog, " | upload %s", n);
            }
        fprintf(gLog, "\n");
    }
    it->second.records.clear();
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f)
{
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < s[i].commandBufferCount; ++k) Flush(s[i].pCommandBuffers[k]);
        ++gSubmits;
        fflush(gLog);
    }
    return r_vkQueueSubmit(q, n, s, f);
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f)
{
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < s[i].commandBufferInfoCount; ++k) Flush(s[i].pCommandBufferInfos[k].commandBuffer);
        ++gSubmits;
        fflush(gLog);
    }
    return r_vkQueueSubmit2(q, n, s, f);
}

static VKAPI_ATTR VkResult VKAPI_CALL H_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi)
{
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        auto& st = gCbs[cb];
        st = CbState{};
    }
    return r_vkBeginCommandBuffer(cb, bi);
}

static void Spoof(VkPhysicalDeviceProperties& p)
{
    if (gSpoof) p.vendorID = 0x1002, p.deviceID = 0x7550;
}

static VKAPI_ATTR void VKAPI_CALL H_vkGetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p)
{
    r_vkGetPhysicalDeviceProperties(pd, p);
    Spoof(*p);
}

static VKAPI_ATTR void VKAPI_CALL H_vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p)
{
    r_vkGetPhysicalDeviceProperties2(pd, p);
    Spoof(p->properties);
    if (gSpoof)
        for (auto* n = (VkBaseOutStructure*)p->pNext; n; n = n->pNext)
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) ((VkPhysicalDeviceDriverProperties*)n)->driverID = VK_DRIVER_ID_AMD_PROPRIETARY;
}

static PFN_vkVoidFunction Hooked(const char* name);

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL H_vkGetDeviceProcAddr(VkDevice d, const char* name)
{
    if (auto h = Hooked(name)) return h;
    return gRealGDPA(d, name);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL H_vkGetInstanceProcAddr(VkInstance i, const char* name)
{
    if (auto h = Hooked(name)) return h;
    return gRealGIPA(i, name);
}

static PFN_vkVoidFunction Hooked(const char* name)
{
#define H(n) if (!strcmp(name, #n)) return (PFN_vkVoidFunction)H_##n;
    H(vkCreateShaderModule) H(vkCreateComputePipelines) H(vkCmdBindPipeline) H(vkCmdBindDescriptorSets) H(vkUpdateDescriptorSets)
    H(vkCmdPushConstants) H(vkCmdDispatch) H(vkCmdDispatchIndirect) H(vkCmdCopyBuffer) H(vkCmdCopyBufferToImage) H(vkCmdUpdateBuffer)
    H(vkCmdFillBuffer) H(vkCreateBuffer) H(vkBindBufferMemory) H(vkMapMemory) H(vkCreateImage) H(vkCreateImageView) H(vkQueueSubmit)
    H(vkQueueSubmit2) H(vkBeginCommandBuffer) H(vkGetPhysicalDeviceProperties) H(vkGetPhysicalDeviceProperties2)
    H(vkCmdPushDescriptorSetKHR) H(vkCmdClearColorImage) H(vkCmdCopyImage) H(vkGetDeviceProcAddr) H(vkGetInstanceProcAddr)
#undef H
    if (!strcmp(name, "vkQueueSubmit2KHR")) return (PFN_vkVoidFunction)H_vkQueueSubmit2;
    return nullptr;
}

// Patches module's import address table entries for vulkan-1.dll with the hooks above.
static int PatchImports(HMODULE mod)
{
    auto base = (uint8_t*)mod;
    auto nt = (IMAGE_NT_HEADERS*)(base + ((IMAGE_DOS_HEADER*)base)->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    int n = 0;
    for (auto imp = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); imp->Name; ++imp)
    {
        if (_stricmp((char*)(base + imp->Name), "vulkan-1.dll")) continue;
        auto names = (IMAGE_THUNK_DATA*)(base + imp->OriginalFirstThunk);
        auto addrs = (IMAGE_THUNK_DATA*)(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++addrs)
        {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            const char* fn = ((IMAGE_IMPORT_BY_NAME*)(base + names->u1.AddressOfData))->Name;
            if (auto h = Hooked(fn))
            {
                DWORD old;
                VirtualProtect(&addrs->u1.Function, sizeof(void*), PAGE_READWRITE, &old);
                addrs->u1.Function = (ULONGLONG)h;
                VirtualProtect(&addrs->u1.Function, sizeof(void*), old, &old);
                ++n;
            }
        }
    }
    return n;
}

// ---------------------------------------------------------------- host side
struct Ctx
{
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t family;
    VkPhysicalDeviceMemoryProperties mp; VkCommandPool pool;
};

static uint32_t MemType(Ctx& c, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (c.mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n"); exit(2);
}

struct Img { VkImage image; VkDeviceMemory mem; VkImageCreateInfo ci; };

static Img MakeImage(Ctx& c, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage)
{
    Img i{};
    i.ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    i.ci.imageType = VK_IMAGE_TYPE_2D, i.ci.format = f, i.ci.extent = {w, h, 1}, i.ci.mipLevels = 1, i.ci.arrayLayers = 1;
    i.ci.samples = VK_SAMPLE_COUNT_1_BIT, i.ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    i.ci.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    CHECK_VK(H_vkCreateImage(c.dev, &i.ci, nullptr, &i.image));  // through the hook so the log names host images too
    VkMemoryRequirements r; vkGetImageMemoryRequirements(c.dev, i.image, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, MemType(c, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CHECK_VK(vkAllocateMemory(c.dev, &ai, nullptr, &i.mem));
    CHECK_VK(vkBindImageMemory(c.dev, i.image, i.mem, 0));
    return i;
}

struct Staging { VkBuffer buf; VkDeviceMemory mem; void* ptr; VkDeviceSize size; };

static Staging MakeStaging(Ctx& c, VkDeviceSize size)
{
    Staging s{}; s.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    CHECK_VK(vkCreateBuffer(c.dev, &bi, nullptr, &s.buf));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.dev, s.buf, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size,
                            MemType(c, r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK_VK(vkAllocateMemory(c.dev, &ai, nullptr, &s.mem));
    CHECK_VK(vkBindBufferMemory(c.dev, s.buf, s.mem, 0));
    CHECK_VK(vkMapMemory(c.dev, s.mem, 0, size, 0, &s.ptr));
    return s;
}

static VkCommandBuffer Begin(Ctx& c)
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, c.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb; CHECK_VK(vkAllocateCommandBuffers(c.dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(H_vkBeginCommandBuffer(cb, &bi));
    return cb;
}

static void End(Ctx& c, VkCommandBuffer cb)
{
    CHECK_VK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    CHECK_VK(H_vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(c.q));
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
}

static void Barrier(VkCommandBuffer cb, VkImage i, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, i, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void Upload(Ctx& c, Img& i, const void* data, size_t bytes, VkImageLayout to)
{
    Staging s = MakeStaging(c, bytes);
    memcpy(s.ptr, data, bytes);
    VkCommandBuffer cb = Begin(c);
    Barrier(cb, i.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = i.ci.extent;
    vkCmdCopyBufferToImage(cb, s.buf, i.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    Barrier(cb, i.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to);
    End(c, cb);
    vkDestroyBuffer(c.dev, s.buf, nullptr); vkFreeMemory(c.dev, s.mem, nullptr);
}

static std::vector<uint8_t> Download(Ctx& c, Img& i, size_t bytes, VkImageLayout from)
{
    Staging s = MakeStaging(c, bytes);
    VkCommandBuffer cb = Begin(c);
    Barrier(cb, i.image, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = i.ci.extent;
    vkCmdCopyImageToBuffer(cb, i.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buf, 1, &r);
    Barrier(cb, i.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, from);
    End(c, cb);
    std::vector<uint8_t> out((uint8_t*)s.ptr, (uint8_t*)s.ptr + bytes);
    vkDestroyBuffer(c.dev, s.buf, nullptr); vkFreeMemory(c.dev, s.mem, nullptr);
    return out;
}

static std::vector<uint8_t> ReadFile(const fs::path& p)
{
    std::vector<uint8_t> v;
    if (FILE* f = fopen(p.string().c_str(), "rb")) { fseek(f, 0, SEEK_END); v.resize(ftell(f)); fseek(f, 0, SEEK_SET); fread(v.data(), 1, v.size(), f); fclose(f); }
    return v;
}

static Ctx MakeDevice(bool validate)
{
    Ctx c{};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "ffx_capture", 1, "nfrux", 1, VK_API_VERSION_1_3};
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, validate ? 1u : 0u, layers};
    CHECK_VK(vkCreateInstance(&ii, nullptr, &c.inst));
    gInstance = c.inst;
    uint32_t n = 0; vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
    c.pd = pds[0];
    for (auto p : pds) { VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr); if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { c.pd = p; break; } }
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(c.pd, &pr);
    printf("device %s vendor %04x driver %u.%u\n", pr.deviceName, pr.vendorID, pr.driverVersion >> 22, (pr.driverVersion >> 14) & 0xff);
    vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);

    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { c.family = i; break; }

    uint32_t en = 0; vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> ext(en); vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &en, ext.data());
    auto has = [&](const char* e) { for (auto& x : ext) if (!strcmp(x.extensionName, e)) return true; return false; };
    std::vector<const char*> want;
    for (const char* e : {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_matrix2", "VK_EXT_shader_float8", "VK_KHR_compute_shader_derivatives",
                          "VK_EXT_shader_image_atomic_int64", "VK_KHR_shader_bfloat16", "VK_KHR_push_descriptor", "VK_KHR_swapchain",
                          "VK_KHR_shader_subgroup_extended_types", "VK_EXT_shader_atomic_float", "VK_KHR_shader_integer_dot_product",
                          "VK_KHR_shader_relaxed_extended_instruction", "VK_KHR_shader_expect_assume"})
        if (has(e)) want.push_back(e);

    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cm2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, has("VK_EXT_shader_float8") ? &f8 : nullptr};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, has("VK_NV_cooperative_matrix2") ? (void*)&cm2 : (void*)(has("VK_EXT_shader_float8") ? &f8 : nullptr)};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR der{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR, has("VK_KHR_cooperative_matrix") ? &cm : nullptr};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT ai64{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT, has("VK_KHR_compute_shader_derivatives") ? (void*)&der : (void*)(has("VK_KHR_cooperative_matrix") ? &cm : nullptr)};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, has("VK_EXT_shader_image_atomic_int64") ? (void*)&ai64 : (void*)&der};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(c.pd, &f2);
    printf("float8 coopmat %d, coopmat2 %d, coopmat %d, float16 %d, int8 %d\n", f8.shaderFloat8CooperativeMatrix, cm2.cooperativeMatrixFlexibleDimensions,
           cm.cooperativeMatrix, v12.shaderFloat16, v12.shaderInt8);
    if (auto props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(c.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"))
    {
        uint32_t m = 0; props(c.pd, &m, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cp(m, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        props(c.pd, &m, cp.data());
        for (auto& p : cp)
            printf("coopmat M%u N%u K%u A%d B%d C%d R%d scope %d\n", p.MSize, p.NSize, p.KSize, int(p.AType), int(p.BType), int(p.CType), int(p.ResultType), int(p.scope));
    }
    // robustness features change behaviour and cost; leave them off as an application normally would
    f2.features.robustBufferAccess = VK_FALSE;

    float prio = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, c.family, 1, &prio};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qi, 0, nullptr, uint32_t(want.size()), want.data()};
    CHECK_VK(vkCreateDevice(c.pd, &di, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.family, 0, &c.q);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, c.family};
    CHECK_VK(vkCreateCommandPool(c.dev, &pi, nullptr, &c.pool));

    gRealGDPA = vkGetDeviceProcAddr; gRealGIPA = vkGetInstanceProcAddr;
#define R(n) r_##n = (PFN_##n)gRealGDPA(c.dev, #n); if (!r_##n) r_##n = (PFN_##n)gRealGIPA(c.inst, #n);
    R(vkCreateShaderModule) R(vkCreateComputePipelines) R(vkCmdBindPipeline) R(vkCmdBindDescriptorSets) R(vkUpdateDescriptorSets)
    R(vkCmdPushConstants) R(vkCmdDispatch) R(vkCmdDispatchIndirect) R(vkCmdCopyBuffer) R(vkCmdCopyBufferToImage) R(vkCmdUpdateBuffer)
    R(vkCmdFillBuffer) R(vkCreateBuffer) R(vkBindBufferMemory) R(vkMapMemory) R(vkCreateImage) R(vkCreateImageView) R(vkQueueSubmit)
    R(vkQueueSubmit2) R(vkBeginCommandBuffer) R(vkCmdPushDescriptorSetKHR) R(vkCmdClearColorImage) R(vkCmdCopyImage)
#undef R
    r_vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    r_vkGetPhysicalDeviceProperties2 = vkGetPhysicalDeviceProperties2;
    return c;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: ffx_capture DLL OUT_DIR [--frames N] [--size WxH] [--spoof-amd] [--validate] [--inputs DIR]\n"); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dll = argv[1];
    gOut = argv[2];
    int frames = 4; uint32_t W = 1920, H = 1080; bool validate = false; std::string inputs;
    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%ux%u", &W, &H);
        else if (!strcmp(argv[i], "--spoof-amd")) gSpoof = true;
        else if (!strcmp(argv[i], "--validate")) validate = true;
        else if (!strcmp(argv[i], "--inputs") && i + 1 < argc) inputs = argv[++i];
        else if (!strcmp(argv[i], "--dump-passes") && i + 3 < argc) gDumpFrame = atoi(argv[++i]), gDumpFirst = atoi(argv[++i]), gDumpLast = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-images") && i + 1 < argc) gImgFrame = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--patch-cb")) gPatchCb = true;
    }
    for (const char* d : {"modules", "uploads", "buffers"}) fs::create_directories(fs::path(gOut) / d);
    gLog = fopen((fs::path(gOut) / "log.txt").string().c_str(), "w");

    Ctx c = MakeDevice(validate);
    gDev = c.dev; gMemProps = c.mp;

    HMODULE mod = LoadLibraryA(dll);
    if (!mod) { fprintf(stderr, "cannot load %s (%lu)\n", dll, GetLastError()); return 2; }
    printf("patched %d vulkan-1.dll imports\n", PatchImports(mod));
    ffxFunctions fx{};
    fx.CreateContext = (PfnFfxCreateContext)GetProcAddress(mod, "ffxCreateContext");
    fx.DestroyContext = (PfnFfxDestroyContext)GetProcAddress(mod, "ffxDestroyContext");
    fx.Configure = (PfnFfxConfigure)GetProcAddress(mod, "ffxConfigure");
    fx.Dispatch = (PfnFfxDispatch)GetProcAddress(mod, "ffxDispatch");
    fx.Query = (PfnFfxQuery)GetProcAddress(mod, "ffxQuery");

    // ---- context
    ffxCreateBackendVKDesc be{};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.deviceContext = {c.dev, c.pd, H_vkGetDeviceProcAddr};
    ffxCreateContextDescFrameGenerationVersion ver{};
    ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    ver.version = FFX_FRAMEGENERATION_VERSION;
    ver.header.pNext = &be.header;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &ver.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    cd.displaySize = {W, H}; cd.maxRenderSize = {W, H};
    cd.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = fx.CreateContext(&ctx, &cd.header, nullptr);
    printf("ffxCreateContext: %u\n", rc);
    if (rc != FFX_API_RETURN_OK) { fclose(gLog); return 3; }

    // ---- resources
    const VkImageUsageFlags storage = VK_IMAGE_USAGE_STORAGE_BIT;
    Img color[2] = {MakeImage(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, storage), MakeImage(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, storage)};
    Img depth = MakeImage(c, W, H, VK_FORMAT_R32_SFLOAT, storage);
    Img mv = MakeImage(c, W, H, VK_FORMAT_R16G16_SFLOAT, storage);
    Img out = MakeImage(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, storage);
    auto Res = [&](Img& i, uint32_t state) { return ffxApiGetResourceVK((void*)i.image, ffxApiGetImageResourceDescriptionVK(i.image, i.ci, 0), state); };

    {
        VkCommandBuffer cb = Begin(c);
        Barrier(cb, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        End(c, cb);
    }

    auto half = [](float f) {
        uint32_t x; memcpy(&x, &f, 4);
        uint32_t s = (x >> 16) & 0x8000; int e = int((x >> 23) & 0xff) - 127 + 15; uint32_t m = (x >> 13) & 0x3ff;
        if (e <= 0) return uint16_t(s);
        if (e >= 31) return uint16_t(s | 0x7c00);
        return uint16_t(s | (e << 10) | m);
    };

    for (int f = 0; f < frames; ++f)
    {
        gFrame = f;
        std::vector<uint8_t> rgba(size_t(W) * H * 4);
        std::vector<float> d(size_t(W) * H);
        std::vector<uint16_t> m(size_t(W) * H * 2);
        bool loaded = false;
        if (!inputs.empty())
        {
            auto a = ReadFile(fs::path(inputs) / ("color_" + std::to_string(f) + ".rgba"));
            auto b = ReadFile(fs::path(inputs) / ("depth_" + std::to_string(f) + ".f32"));
            auto v = ReadFile(fs::path(inputs) / ("mv_" + std::to_string(f) + ".rg32f"));
            if (a.size() == rgba.size() && b.size() == d.size() * 4 && v.size() == m.size() * 4)
            {
                rgba = a; memcpy(d.data(), b.data(), b.size());
                const float* fv = (const float*)v.data();
                for (size_t i = 0; i < m.size(); ++i) m[i] = half(fv[i]);
                loaded = true;
            }
        }
        if (!loaded)
        {
            const int shift = 8 * f;
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                {
                    int u = int(x) - shift;
                    uint8_t* p = &rgba[(size_t(y) * W + x) * 4];
                    p[0] = uint8_t(128 + 100 * sinf(u * 0.05f) * cosf(y * 0.031f));
                    p[1] = uint8_t(((u >> 4) ^ (int(y) >> 4)) & 1 ? 200 : 40);
                    p[2] = uint8_t((u * 3 + y) & 255);
                    p[3] = 255;
                    d[size_t(y) * W + x] = 0.1f;
                    m[(size_t(y) * W + x) * 2 + 0] = half(-8.0f);  // current -> previous, pixels
                    m[(size_t(y) * W + x) * 2 + 1] = half(0.0f);
                }
        }
        Img& cur = color[f & 1];
        Upload(c, cur, rgba.data(), rgba.size(), VK_IMAGE_LAYOUT_GENERAL);
        Upload(c, depth, d.data(), d.size() * 4, VK_IMAGE_LAYOUT_GENERAL);
        Upload(c, mv, m.data(), m.size() * 2, VK_IMAGE_LAYOUT_GENERAL);

        // configure (no swapchain: generation only)
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.frameGenerationEnabled = true;
        cfg.flags = 0;
        cfg.generationRect = {0, 0, int32_t(W), int32_t(H)};
        cfg.frameID = uint64_t(f);
        rc = fx.Configure(&ctx, &cfg.header);
        if (rc != FFX_API_RETURN_OK) printf("frame %d configure: %u\n", f, rc);

        VkCommandBuffer cb = Begin(c);
        ffxDispatchDescFrameGenerationPrepareV2 pre{};
        pre.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
        pre.frameID = uint64_t(f);
        pre.commandList = cb;
        pre.renderSize = {W, H};
        pre.jitterOffset = {0, 0};
        pre.motionVectorScale = {1, 1};
        pre.frameTimeDelta = 33.3f;
        pre.reset = f == 0;
        pre.cameraNear = 0.1f; pre.cameraFar = 1000.f; pre.cameraFovAngleVertical = 1.0f; pre.viewSpaceToMetersFactor = 1;
        pre.depth = Res(depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        pre.motionVectors = Res(mv, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        float cp[3] = {0, 0, 0}, up[3] = {0, 1, 0}, rt[3] = {1, 0, 0}, fw[3] = {0, 0, 1};
        memcpy(pre.cameraPosition, cp, 12); memcpy(pre.cameraUp, up, 12); memcpy(pre.cameraRight, rt, 12); memcpy(pre.cameraForward, fw, 12);
        rc = fx.Dispatch(&ctx, &pre.header);
        if (rc != FFX_API_RETURN_OK) printf("frame %d prepare: %u\n", f, rc);

        ffxDispatchDescFrameGeneration fg{};
        fg.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        fg.commandList = cb;
        fg.presentColor = Res(cur, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        fg.outputs[0] = Res(out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        fg.numGeneratedFrames = 1;
        fg.reset = f == 0;
        fg.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
        fg.minMaxLuminance[0] = 0; fg.minMaxLuminance[1] = 1000;
        fg.generationRect = {0, 0, int32_t(W), int32_t(H)};
        fg.frameID = uint64_t(f);
        rc = fx.Dispatch(&ctx, &fg.header);
        if (rc != FFX_API_RETURN_OK) printf("frame %d generate: %u\n", f, rc);
        End(c, cb);
        fprintf(gLog, "FRAME %d done\n", f);
        if (f == gDumpFrame) WritePassDumps();
        if (f == gImgFrame) WriteImageDumps();

        auto px = Download(c, out, size_t(W) * H * 4, VK_IMAGE_LAYOUT_GENERAL);
        FILE* o = fopen((fs::path(gOut) / ("output_" + std::to_string(f) + ".rgba")).string().c_str(), "wb");
        fwrite(px.data(), 1, px.size(), o); fclose(o);
        double sum = 0; for (size_t i = 0; i < px.size(); i += 4) sum += px[i];
        printf("frame %d: output mean red %.2f\n", f, sum / (W * H));
    }

    // ---- final buffer contents
    {
        std::lock_guard<std::recursive_mutex> l(gLock);
        std::vector<std::pair<VkBuffer, BufferInfo>> bufs(gBuffers.begin(), gBuffers.end());
        for (auto& [b, bi] : bufs)
        {
            if (bi.size < 65536 || bi.size > (512ull << 20) || !bi.mem) continue;
            Staging s = MakeStaging(c, bi.size);
            VkCommandBuffer cb = Begin(c);
            VkBufferCopy r{0, 0, bi.size};
            r_vkCmdCopyBuffer(cb, b, s.buf, 1, &r);
            CHECK_VK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1, si.pCommandBuffers = &cb;
            CHECK_VK(r_vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(c.q));
            char n[32]; snprintf(n, sizeof(n), "%u.bin", bi.id);
            FILE* f = fopen((fs::path(gOut) / "buffers" / n).string().c_str(), "wb");
            fwrite(s.ptr, 1, size_t(bi.size), f); fclose(f);
            fprintf(gLog, "BUF buf%u size %llu usage %x\n", bi.id, (unsigned long long)bi.size, bi.usage);
            vkDestroyBuffer(c.dev, s.buf, nullptr); vkFreeMemory(c.dev, s.mem, nullptr);
        }
    }
    fx.DestroyContext(&ctx, nullptr);
    fclose(gLog);
    printf("captured %llu commands in %u submits, %u uploads\n", (unsigned long long)gSeq, gSubmits, gUploads);
    return 0;
}
