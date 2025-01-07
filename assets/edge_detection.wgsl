#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

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
    debug: u32,
    enabled: u32,
}

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

fn load_view_z(frag_coord: vec2i) -> f32 {
    let depth = textureLoad(depth_prepass_texture, frag_coord, 0);
    return depth_ndc_to_view_z(depth);
}

fn view_z_gradient_x(frag_coord: vec2i, y: i32) -> f32 {
    let l_coord = frag_coord + vec2i(-1, y);    // left  coordinate
    let r_coord = frag_coord + vec2i( 1, y);    // right coordinate

    return load_view_z(r_coord) - load_view_z(l_coord); 
}

fn view_z_gradient_y(frag_coord: vec2i, x: i32) -> f32 {
    let d_coord = frag_coord + vec2i(x, -1);    // down coordinate
    let t_coord = frag_coord + vec2i(x,  1);    // top  coordinate

    return load_view_z(t_coord) - load_view_z(d_coord);
}

fn detect_edge_depth(frag_coord: vec2i) -> f32 {
    if ed_uniform.depth_threshold == 0.0 { return 0.0; }

    let grad_x = 
        view_z_gradient_x(frag_coord,  1) +
        2.0 * view_z_gradient_x(frag_coord,  0) +
        view_z_gradient_x(frag_coord, -1);

    let grad_y =
        view_z_gradient_y(frag_coord, 1) +
        2.0 * view_z_gradient_y(frag_coord, 0) +
        view_z_gradient_y(frag_coord, -1);

    let grad = max(abs(grad_x), abs(grad_y));

    return select(0.0, grad, grad > 2.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = vec2i(in.position.xy);

    let color = textureSample(screen_texture, texture_sampler, in.uv).rgb;

    let normal = textureLoad(normal_prepass_texture, frag_coord, 0).rgb;
    let depth = textureLoad(depth_prepass_texture, frag_coord, 0);
    
    let edge_depth = detect_edge_depth(frag_coord);

    let final_color = select(vec4f(color, 1.0), ed_uniform.edge_color, edge_depth > 0.01);
    // return vec4f(vec3f(edge_depth), 1.0);
    return final_color;
}

