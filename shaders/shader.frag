#version 450
	

layout (location=0) out vec4 theColour;

layout (location=0) flat in vec4 data_from_the_vertexshader;

void main(){
	theColour= data_from_the_vertexshader;
}