// The network's layers at a size: shapes, multiply-adds and int8 bytes moved, for judging the kernels against the GPU's peak.
//   model_layers [W H]   (the network's own size, as the interpolator gives it; default 992 544)   SPDX-License-Identifier: MIT
#include "model.h"
#include "tosa_graph.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    const uint32_t w = argc > 2 ? uint32_t(atoi(argv[1])) : 992, h = argc > 2 ? uint32_t(atoi(argv[2])) : 544;
    ffx_ml::tosa::Graph  g;
    ffx_ml::LoweredGraph l;
    std::string          e;
    if (!ffx_ml::tosa::Parse(nfrux::DescribeModel(w, h), g, &e) || !ffx_ml::Lower(g, l, &e)) return printf("%s\n", e.c_str()), 1;
    double macs = 0, bytes = 0;
    printf(" #  k s   in HxWxC          out HxWxC        MMAC    MB\n");
    for (size_t i = 0; i < l.convs.size(); ++i)
    {
        const auto& c = l.convs[i];
        const double m = double(c.outH) * c.outW * c.outC * c.inC * c.conv->kernelH * c.conv->kernelW, b = double(c.inH) * c.inW * c.inC + double(c.outH) * c.outW * c.outC;
        macs += m, bytes += b;
        printf("%2zu  %d %d  %4dx%4dx%-3d  -> %4dx%4dx%-3d  %7.1f  %5.2f\n", i, c.conv->kernelH, c.conv->stride[0], c.inH, c.inW, c.inC, c.outH, c.outW, c.outC, m / 1e6, b / 1e6);
    }
    printf("%zu layers: %.2f GMAC, %.1f MB of int8 activations read and written (weights not counted)\n", l.convs.size(), macs / 1e9, bytes / 1e6);
}
