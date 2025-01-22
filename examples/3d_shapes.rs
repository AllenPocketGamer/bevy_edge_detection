//! This example comes from [3d_shapes](https://github.com/bevyengine/bevy/blob/main/examples/3d/3d_shapes.rs)

use std::f32::consts::PI;

use bevy::{
    color::palettes::basic::SILVER,
    core_pipeline::{core_3d::graph::Node3d, smaa::Smaa},
    input::common_conditions::input_toggle_active,
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
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
        .add_systems(Startup, (setup, spawn_text))
        .add_systems(
            Update,
            (
                rotate.run_if(input_toggle_active(false, KeyCode::Space)),
                edge_detection_ui,
            ),
        )
        .run();
}

/// A marker component for our shapes so we can query them separately from the ground plane
#[derive(Component)]
struct Shape;

const SHAPES_X_EXTENT: f32 = 14.0;
const EXTRUSION_X_EXTENT: f32 = 16.0;
const Z_EXTENT: f32 = 5.0;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let debug_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });

    let shapes = [
        meshes.add(Cuboid::default()),
        meshes.add(Tetrahedron::default()),
        meshes.add(Capsule3d::default()),
        meshes.add(Torus::default()),
        meshes.add(Cylinder::default()),
        meshes.add(Cone::default()),
        meshes.add(ConicalFrustum::default()),
        meshes.add(Sphere::default().mesh().ico(5).unwrap()),
        meshes.add(Sphere::default().mesh().uv(32, 18)),
    ];

    let extrusions = [
        meshes.add(Extrusion::new(Rectangle::default(), 1.)),
        meshes.add(Extrusion::new(Capsule2d::default(), 1.)),
        meshes.add(Extrusion::new(Annulus::default(), 1.)),
        meshes.add(Extrusion::new(Circle::default(), 1.)),
        meshes.add(Extrusion::new(Ellipse::default(), 1.)),
        meshes.add(Extrusion::new(RegularPolygon::default(), 1.)),
        meshes.add(Extrusion::new(Triangle2d::default(), 1.)),
    ];

    let num_shapes = shapes.len();

    for (i, shape) in shapes.into_iter().enumerate() {
        commands.spawn((
            Mesh3d(shape),
            MeshMaterial3d(debug_material.clone()),
            Transform::from_xyz(
                -SHAPES_X_EXTENT / 2. + i as f32 / (num_shapes - 1) as f32 * SHAPES_X_EXTENT,
                2.0,
                Z_EXTENT / 2.,
            )
            .with_rotation(Quat::from_rotation_x(-PI / 4.)),
            Shape,
        ));
    }

    let num_extrusions = extrusions.len();

    for (i, shape) in extrusions.into_iter().enumerate() {
        commands.spawn((
            Mesh3d(shape),
            MeshMaterial3d(debug_material.clone()),
            Transform::from_xyz(
                -EXTRUSION_X_EXTENT / 2.
                    + i as f32 / (num_extrusions - 1) as f32 * EXTRUSION_X_EXTENT,
                2.0,
                -Z_EXTENT / 2.,
            )
            .with_rotation(Quat::from_rotation_x(-PI / 4.)),
            Shape,
        ));
    }

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

    // ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0).subdivisions(10))),
        MeshMaterial3d(materials.add(Color::from(SILVER))),
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

fn spawn_text(mut commands: Commands) {
    commands.spawn((
        Text::new("Press Space to turn on/off rotation!"),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn rotate(mut query: Query<&mut Transform, With<Shape>>, time: Res<Time>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_secs() / 2.);
    }
}

/// Creates a colorful test pattern
fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
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

            let mut color = edge_detection.edge_color.to_srgba().to_u8_array_no_alpha();
            ui.horizontal(|ui| {
                egui::color_picker::color_edit_button_srgb(ui, &mut color);
                ui.label("edge_color");
            });
            edge_detection.edge_color = Color::srgb_u8(color[0], color[1], color[2]);
        });
    });
}
