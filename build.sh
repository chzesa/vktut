#!/bin/sh
rm VulkanTest
glslangValidator -V shader.vert -V shader.frag
make