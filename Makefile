STB_INCLUDE_PATH = ./extern/stb
TOBJ_INCLUDE_PATH = ./extern/tinyobjloader

LDFLAGS = `pkg-config --static --libs glfw3 vulkan`

VulkanTest: main.cpp
	g++ -std=c++17 -I$(STB_INCLUDE_PATH) -I$(TOBJ_INCLUDE_PATH) -o VulkanTest main.cpp $(LDFLAGS)

clean:
	rm -f VulkanTest