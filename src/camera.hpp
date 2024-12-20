#include "types.h"

class Camera {
public:
    struct GLFWwindow* window;
    
    glm::vec3 velocity = glm::vec3(0);
    glm::vec3 position = glm::vec3(0);
    // vertical rotation
    float pitch { 0.f };
    // horizontal rotation
    float yaw { 0.f };

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();
    glm::mat4 getLookAtMatrix(glm::vec3 target, glm::vec3 upVector);
    
    void getSixAxis();

    void processEvent();

    void update();
};


