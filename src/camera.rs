extern crate nalgebra as na;
use gpu_allocator::vulkan::*;

use crate::GpuBuffer;

pub struct Camera {
    pub view_matrix: na::Matrix4<f32>,
    position: na::Vector3<f32>,
    view_direction: na::Unit<na::Vector3<f32>>,
    down_direction: na::Unit<na::Vector3<f32>>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            view_matrix: na::Matrix4::identity(),
            position: na::Vector3::new(0.0, 0.0, 0.0),
            view_direction: na::Unit::new_normalize(na::Vector3::new(0.0, 0.0, 1.0)),
            down_direction: na::Unit::new_normalize(na::Vector3::new(0.0, 1.0, 0.0)),
        }
    }
}

impl Camera {
    pub fn update_buffer(&self, allocator: &mut Allocator, buffer: &mut GpuBuffer) {
        let data: [[f32; 4]; 4] = self.view_matrix.into();
        buffer.write_to_memory(allocator, &data).unwrap();
    }

    fn update_view_matrix(&mut self) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let m = na::Matrix4::new(
            right.x,
            right.y,
            right.z,
            -right.dot(&self.position), //
            self.down_direction.x,
            self.down_direction.y,
            self.down_direction.z,
            -self.down_direction.dot(&self.position), //
            self.view_direction.x,
            self.view_direction.y,
            self.view_direction.z,
            -self.view_direction.dot(&self.position), //
            0.0,
            0.0,
            0.0,
            1.0,
        );
        self.view_matrix = m;
    }

    pub fn move_forward(&mut self, distance: f32) {
        self.position += distance * self.view_direction.as_ref();
        self.update_view_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }
    pub fn turn_right(&mut self, angle: f32) {
        let rotation = na::Rotation3::from_axis_angle(&self.down_direction, angle);
        self.view_direction = rotation * self.view_direction;
        self.update_view_matrix();
    }
    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let rotation = na::Rotation3::from_axis_angle(&right, angle);
        self.view_direction = rotation * self.view_direction;
        self.down_direction = rotation * self.down_direction;
        self.update_view_matrix();
    }
    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }
}
