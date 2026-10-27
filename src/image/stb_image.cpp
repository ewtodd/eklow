// stb_image's implementation, limited to the formats clients send: fewer parsers facing request
// bytes than llama.cpp's build, which also reads TGA, PSD, HDR, PIC and PNM.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
#define STBI_NO_STDIO
#include "stb/stb_image.h"
