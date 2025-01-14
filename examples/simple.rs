//! This example comes from [3d_shapes](https://github.com/bevyengine/bevy/blob/main/examples/3d/3d_shapes.rs)

use bevy::{
    core_pipeline::{core_3d::graph::Node3d, smaa::Smaa},
    prelude::*,
};
use bevy_edge_detection::{EdgeDetection, EdgeDetectionPlugin};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugins(EdgeDetectionPlugin {
            // If you wish to apply Smaa anti-aliasing after edge detection,
            // please ensure that the rendering order of [`EdgeDetectionNode`] is set before [`SmaaNode`].
            before: Node3d::Smaa,
        })
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, edge_detection_ui)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_length(8.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.6509, 0.6509, 0.6509),
            unlit: true,
            ..default()
        })),
        Transform::from_scale(Vec3::new(1.0, 0.5, 1.0)),
    ));

    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 10_000_000.,
            range: 100.0,
            shadow_depth_bias: 0.2,
            ..default()
        },
        Transform::from_xyz(8.0, 16.0, 8.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
        Camera {
            clear_color: Color::WHITE.into(),
            ..default()
        },
        // [`EdgeDetectionNode`] supports `Msaa``, and you can enable it at any time, for example:
        // Msaa::default(),
        Msaa::Off,
        EdgeDetection::default(),
        Smaa::default(),
        // to control camera
        PanOrbitCamera::default(),
    ));
}

fn edge_detection_ui(mut ctx: EguiContexts, mut edge_detection: Single<&mut EdgeDetection>) {
    egui::Window::new("Edge Detection Settings").show(ctx.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.add(egui::Checkbox::new(
                    &mut edge_detection.enable_depth,
                    "enable_depth",
                ));
                ui.add(
                    egui::Slider::new(&mut edge_detection.depth_threshold, 0.0..=8.0)
                        .text("depth_threshold"),
                );
            });

            ui.horizontal(|ui| {
                ui.add(egui::Checkbox::new(
                    &mut edge_detection.enable_normal,
                    "enable_normal",
                ));
                ui.add(
                    egui::Slider::new(&mut edge_detection.normal_threshold, 0.0..=8.0)
                        .text("normal_threshold"),
                );
            });

            ui.horizontal(|ui| {
                ui.add(egui::Checkbox::new(
                    &mut edge_detection.enable_color,
                    "enable_color",
                ));
                ui.add(
                    egui::Slider::new(&mut edge_detection.color_threshold, 0.0..=8.0)
                        .text("color_threshold"),
                );
            });

            ui.add(
                egui::Slider::new(&mut edge_detection.depth_thickness, 0.0..=8.0)
                    .text("depth_thickness"),
            );
            ui.add(
                egui::Slider::new(&mut edge_detection.normal_thickness, 0.0..=8.0)
                    .text("normal_thickness"),
            );
            ui.add(
                egui::Slider::new(&mut edge_detection.color_thickness, 0.0..=8.0)
                    .text("color_thickness"),
            );

            ui.add(
                egui::Slider::new(&mut edge_detection.steep_angle_threshold, 0.0..=1.0)
                    .text("steep_angle_threshold"),
            );
            ui.add(
                egui::Slider::new(&mut edge_detection.steep_angle_multiplier, 0.0..=1.0)
                    .text("steep_angle_multiplier"),
            );

            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut edge_detection.uv_distortion_frequency.x)
                        .range(0.0..=16.0),
                );
                ui.add(
                    egui::DragValue::new(&mut edge_detection.uv_distortion_frequency.y)
                        .range(0.0..=16.0),
                );
                ui.label("uv_distortion_frequency");
            });

            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut edge_detection.uv_distortion_strength.x)
                        .range(0.0..=1.0)
                        .fixed_decimals(4),
                );
                ui.add(
                    egui::DragValue::new(&mut edge_detection.uv_distortion_strength.y)
                        .range(0.0..=1.0)
                        .fixed_decimals(4),
                );
                ui.label("uv_distortion_strength");
            });

            let mut color = edge_detection.edge_color.to_srgba().to_f32_array_no_alpha();
            ui.horizontal(|ui| {
                egui::color_picker::color_edit_button_rgb(ui, &mut color);
                ui.label("edge_color");
            });
            edge_detection.edge_color = Color::srgb_from_array(color);
        });
    });
}
