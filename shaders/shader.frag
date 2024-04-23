#version 450


layout (location=0) out vec4 theColour;

layout (location=0) in vec4 data_from_the_vertexshader;
layout (location=1) in vec3 normal;

void main(){
	vec3 direction_to_light=normalize(vec3(-1,-1,0));
	theColour= 0.5*(1+max(dot(normal,direction_to_light),0))*data_from_the_vertexshader;
}