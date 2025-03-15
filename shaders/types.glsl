struct VertexData{
    highp vec3 pos;
    float invMass;
};

struct VertexDynamics{
    highp vec3 velocity;
    float invMass;
};

struct corrStruct{
    highp vec3 dP;
    float pad;
};

