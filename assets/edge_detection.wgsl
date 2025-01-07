#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import bevy_pbr::view_transformations::uv_to_ndc

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(3) var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> view: View;
@group(0) @binding(5) var<uniform> ed_uniform: EdgeDetectionUniform;

struct EdgeDetectionUniform {
    depth_threshold: f32,
    normal_threshold: f32,
    color_threshold: f32,
    edge_color: vec4f,

    steep_angle_threshold: f32,
    steep_angle_multiplier: f32,
}

// -----------------------
// View Transformation ---
// -----------------------

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near() -> f32 {
    return view.clip_from_view[3][2];
}

/// Convert ndc depth to linear view z. 
/// Note: Depth values in front of the camera will be negative as -z is forward
fn depth_ndc_to_view_z(ndc_depth: f32) -> f32 {
#ifdef VIEW_PROJECTION_PERSPECTIVE
    return -perspective_camera_near() / ndc_depth;
#else ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    return -(view.clip_from_view[3][2] - ndc_depth) / view.clip_from_view[2][2];
#else
    let view_pos = view.view_from_clip * vec4(0.0, 0.0, ndc_depth, 1.0);
    return view_pos.z / view_pos.w;
#endif
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3f) -> vec3f {
    let world_pos = view.world_from_clip * vec4f(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert a view space direction to world space
fn direction_view_to_world(view_dir: vec3<f32>) -> vec3<f32> {
    let world_dir = view.world_from_view * vec4(view_dir, 0.0);
    return world_dir.xyz;
}

fn calculate_view(world_position: vec3f) -> vec3f {
#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
        // Orthographic view vector
        return normalize(vec3f(view.clip_from_world[0].z, view.clip_from_world[1].z, view.clip_from_world[2].z));
#else
        // Only valid for a perspective projection
        return normalize(view.world_position.xyz - world_position.xyz);
#endif
}

// -----------------------
// Depth Detection -------
// -----------------------

fn prepass_view_z(pixel_coord: vec2i) -> f32 {
    let depth = textureLoad(depth_prepass_texture, pixel_coord, 0);
    return depth_ndc_to_view_z(depth);
}

fn view_z_gradient_x(pixel_coord: vec2i, y: i32) -> f32 {
    let l_coord = pixel_coord + vec2i(-1, y);    // left  coordinate
    let r_coord = pixel_coord + vec2i( 1, y);    // right coordinate

    return prepass_view_z(r_coord) - prepass_view_z(l_coord); 
}

fn view_z_gradient_y(pixel_coord: vec2i, x: i32) -> f32 {
    let d_coord = pixel_coord + vec2i(x, -1);    // down coordinate
    let t_coord = pixel_coord + vec2i(x,  1);    // top  coordinate

    return prepass_view_z(t_coord) - prepass_view_z(d_coord);
}

fn detect_edge_depth(pixel_coord: vec2i, NdotV: f32) -> f32 {
    if ed_uniform.depth_threshold == 0.0 { return 0.0; }

    let grad_x = 
        view_z_gradient_x(pixel_coord,  1) +
        2.0 * view_z_gradient_x(pixel_coord,  0) +
        view_z_gradient_x(pixel_coord, -1);

    let grad_y =
        view_z_gradient_y(pixel_coord, 1) +
        2.0 * view_z_gradient_y(pixel_coord, 0) +
        view_z_gradient_y(pixel_coord, -1);

    // why not `let grad = sqrt(grad_x * grad_x + grad_y * grad_y);`?
    //
    // Because ·grad_x· or ·grad_y· might be too large,
    // causing overflow in the calculation and resulting in incorrect results.
    let grad = max(abs(grad_x), abs(grad_y));

    let steep_angle_adjustment = 
        smoothstep(ed_uniform.steep_angle_threshold, 1.0, 1.0 - NdotV) * ed_uniform.steep_angle_multiplier + 1.0;

    return f32(grad > ed_uniform.depth_threshold * steep_angle_adjustment);
}

// -----------------------
// Normal Detection ------
// -----------------------

fn prepass_normal(pixel_coord: vec2i) -> vec3f {
#ifdef MULTISAMPLED
    let normal_sample = textureLoad(normal_prepass_texture, pixel_coord, 0);
#else
    let normal_sample = textureLoad(normal_prepass_texture, pixel_coord, 0);
#endif // MULTISAMPLED
    return normalize(normal_sample.xyz * 2.0 - vec3(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(screen_texture, texture_sampler, in.uv).rgb;

    let pixel_coord = vec2i(in.position.xy);

    let ndc = vec3f(uv_to_ndc(in.uv), 1.0);
    let world_position = position_ndc_to_world(ndc);

    let view_direction = calculate_view(world_position);
    let normal = prepass_normal(pixel_coord);
    let NdotV = max(dot(normal, view_direction), 0.001);

    let edge_depth = detect_edge_depth(pixel_coord, NdotV);

    color = mix(color, ed_uniform.edge_color.rgb, edge_depth);

    return vec4f(color, 1.0);
}