use bevy::{
    core_pipeline::prepass::{DepthPrepass, NormalPrepass},
    prelude::*,
};
use bevy_edge_detection::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, EdgeDetectionPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate, update_settings))
        .run();
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::default(), Vec3::Y),
        Camera {
            clear_color: Color::WHITE.into(),
            ..default()
        },
        // TODO: [`EdgeDetectionPlugin`] not support `MSAA` currently.
        Msaa::Off,
        // The edge detection effect requires the depth prepass
        DepthPrepass,
        // The edge detection effect requires the normal prepass
        NormalPrepass,
        // The edge detection settings. This component is also used to determine on which camera to run the edge detection post-processing.
        EdgeDetectionUniform {
            intensity: 0.02,
            ..default()
        },
    ));

    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Rotates,
    ));
    // light
    commands.spawn(DirectionalLight {
        illuminance: 1_000.,
        ..default()
    });
}

#[derive(Component)]
struct Rotates;

/// Rotates any entity around the x and y axis
fn rotate(time: Res<Time>, mut query: Query<&mut Transform, With<Rotates>>) {
    for mut transform in &mut query {
        transform.rotate_x(0.55 * time.delta_secs());
        transform.rotate_z(0.15 * time.delta_secs());
    }
}

// Change the intensity over time to show that the effect is controlled from the main world
fn update_settings(mut settings: Query<&mut EdgeDetectionUniform>, time: Res<Time>) {
    for mut setting in &mut settings {
        let mut intensity = ops::sin(time.elapsed_secs());
        // Make it loop periodically
        intensity = ops::sin(intensity);
        // Remap it to 0..1 because the intensity can't be negative
        intensity = intensity * 0.5 + 0.5;
        // Scale it to a more reasonable level
        intensity *= 0.015;

        // Set the intensity.
        // This will then be extracted to the render world and uploaded to the GPU automatically by the [`UniformComponentPlugin`]
        setting.intensity = intensity;
    }
}
