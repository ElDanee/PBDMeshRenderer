#include "camera.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <GLFW/glfw3.h>

void Camera::getSixAxis() {
    const float MOUSE_RES = 200.0f;
    static double old_xpos = 0, old_ypos = 0;
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    double m_dx = xpos - old_xpos;
    double m_dy = ypos - old_ypos;
    old_xpos = xpos; old_ypos = ypos;
    
    if(!glfwGetKey(window, GLFW_KEY_C))
    {
        yaw += m_dx / MOUSE_RES;
        pitch += -m_dy / MOUSE_RES;
    }

    if(glfwGetKey(window, GLFW_KEY_A)) {
        velocity.x = -1.0f;
    }
    else if(glfwGetKey(window, GLFW_KEY_D)) {
        velocity.x = 1.0f;
    } else velocity.x = .0f;
    
    if(glfwGetKey(window, GLFW_KEY_S)) {
        velocity.z = 1.0f;
    }
    else if(glfwGetKey(window, GLFW_KEY_W)) {
        velocity.z = -1.0f;
    } else velocity.z = .0f;
}

void Camera::update()
{
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity/3.f, 0.f));
}

void Camera::processEvent()
{
    getSixAxis();
}

glm::mat4 Camera::getViewMatrix()
{
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    // so we will create the camera model matrix and invert
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix()
{
    // FPS style camera.
    // join the pitch and yaw rotations into the final rotation matrix
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3 { 1.f, 0.f, 0.f });
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3 { 0.f, -1.f, 0.f });

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

glm::mat4 Camera::getLookAtMatrix(glm::vec3 target, glm::vec3 upVector){
    return glm::lookAt(position, target, upVector);
}

