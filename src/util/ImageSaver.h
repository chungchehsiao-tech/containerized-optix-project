#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h> // for uchar4

inline void savePPM(const char* filename, const std::vector<uchar4>& pixels, int width, int height) {
    // To guarantee zero external dependencies, we use the raw PPM image format.
    std::cout << "Saving image to output.ppm...\n";
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n"; // PPM Header
    for (int i = 0; i < width * height; ++i) {
        file << (int)pixels[i].x << " " 
             << (int)pixels[i].y << " " 
             << (int)pixels[i].z << "\n";
    }
    file.close();
    std::cout << "[SUCCESS] Image successfully rendered and saved to the project folder!\n";
}