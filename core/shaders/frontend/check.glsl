// The self-check's input tensor: hashed words (the front end makes the same on the CPU). SPDX-License-Identifier: MIT
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) buffer Tensor { uint data[]; } t;
void main()
{
    const uint i = gl_GlobalInvocationID.x;
    uint h = i * 2654435761u + 1013904223u;
    h = (h ^ (h >> 15)) * 2246822519u;
    h = (h ^ (h >> 13)) * 3266489917u;
    t.data[i] = h ^ (h >> 16);
}