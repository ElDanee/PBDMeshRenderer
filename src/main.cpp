#include "engine.hpp"
#include "stdlib.h"

int main(int argc, char* argv[])
{
    VulkanEngine engine;
    
    int subdivisions = 32;
    if(argc>1){
        subdivisions = (int)std::abs(strtol(argv[1], NULL, 10));
    }
    
    int iters = 20;
    if(argc>2){
        iters = (int)std::abs(strtol(argv[2], NULL, 10));
    }
    
    bool use3D = 1;
    if(argc>3){
        use3D = (int)std::abs(strtol(argv[3], NULL, 10));
    }
    
    engine.init(subdivisions, iters, use3D);
    
    engine.run();

    engine.cleanup();

    return 0;
}
