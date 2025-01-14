//! Edge Detection using 3x3 Sobel Filter
//!
//! This shader implements edge detection based on depth, normal, and color gradients using a 3x3 Sobel filter.
//! It combines the results of depth, normal, and color edge detection to produce a final edge map.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import bevy_pbr::view_transformations::uv_to_ndc

@group(0) @binding(0) var screen_texture: texture_2d<f32>;

#ifdef MULTISAMPLED
@group(0) @binding(1) var depth_prepass_texture: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth_prepass_texture: texture_depth_2d;
#endif

#ifdef MULTISAMPLED
@group(0) @binding(2) var normal_prepass_texture: texture_multisampled_2d<f32>;
#else
@group(0) @binding(2) var normal_prepass_texture: texture_2d<f32>;
#endif

@group(0) @binding(3) var texture_sampler: sampler;
@group(0) @binding(4) var<uniform> view: View;
@group(0) @binding(5) var<uniform> ed_uniform: EdgeDetectionUniform;

struct EdgeDetectionUniform {
    depth_threshold: f32,
    normal_threshold: f32,
    color_threshold: f32,

    depth_thickness: f32,
    normal_thickness: f32,
    color_thickness: f32,
    
    steep_angle_threshold: f32,
    steep_angle_multiplier: f32,

    edge_color: vec4f,
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

fn prepass_view_z(uv: vec2f) -> f32 {
#ifdef MULTISAMPLED
    let pixel_coord = vec2i(uv * texture_size);
    let depth = textureLoad(depth_prepass_texture, pixel_coord, sample_index_i);
#else
    let depth = textureSample(depth_prepass_texture, texture_sampler, uv);
#endif
    return depth_ndc_to_view_z(depth);
}

fn view_z_gradient_x(uv: vec2f, y: f32, thickness: f32) -> f32 {
    let l_coord = uv + texel_size * vec2f(-thickness, y);    // left  coordinate
    let r_coord = uv + texel_size * vec2f( thickness, y);    // right coordinate

    return prepass_view_z(r_coord) - prepass_view_z(l_coord); 
}

fn view_z_gradient_y(uv: vec2f, x: f32, thickness: f32) -> f32 {
    let d_coord = uv + texel_size * vec2f(x, -thickness);    // down coordinate
    let t_coord = uv + texel_size * vec2f(x,  thickness);    // top  coordinate

    return prepass_view_z(t_coord) - prepass_view_z(d_coord);
}

fn detect_edge_depth(uv: vec2f, thickness: f32, fresnel: f32) -> f32 {
    let deri_x = 
        view_z_gradient_x(uv, thickness, thickness) +
        2.0 * view_z_gradient_x(uv, 0.0, thickness) +
        view_z_gradient_x(uv, -thickness, thickness);

    let deri_y =
        view_z_gradient_y(uv, thickness, thickness) +
        2.0 * view_z_gradient_y(uv, 0.0, thickness) +
        view_z_gradient_y(uv, -thickness, thickness);

    // why not `let grad = sqrt(deri_x * deri_x + deri_y * deri_y);`?
    //
    // Because ·deri_x· or ·deri_y· might be too large,
    // causing overflow in the calculation and resulting in incorrect results.
    let grad = max(abs(deri_x), abs(deri_y));

    let view_z = abs(prepass_view_z(uv));

    let steep_angle_adjustment = 
        smoothstep(ed_uniform.steep_angle_threshold, 1.0, fresnel) * ed_uniform.steep_angle_multiplier * view_z;

    return f32(grad > ed_uniform.depth_threshold * (1.0 + steep_angle_adjustment));
}

// -----------------------
// Normal Detection ------
// -----------------------

fn prepass_normal_unpack(uv: vec2f) -> vec3f {
    let normal_packed = prepass_normal(uv);
    return normalize(normal_packed.xyz * 2.0 - vec3(1.0));
}

fn prepass_normal(uv: vec2f) -> vec3f {
#ifdef MULTISAMPLED
    let pixel_coord = vec2i(uv * texture_size);
    let normal = textureLoad(normal_prepass_texture, pixel_coord, sample_index_i);
#else
    let normal = textureSample(normal_prepass_texture, texture_sampler, uv);
#endif
    return normal.xyz;
}

fn normal_gradient_x(uv: vec2f, y: f32, thickness: f32) -> vec3f {
    let l_coord = uv + texel_size * vec2f(-thickness, y);    // left  coordinate
    let r_coord = uv + texel_size * vec2f( thickness, y);    // right coordinate

    return prepass_normal(r_coord) - prepass_normal(l_coord);
}

fn normal_gradient_y(uv: vec2f, x: f32, thickness: f32) -> vec3f {
    let d_coord = uv + texel_size * vec2f(x, -thickness);    // down coordinate
    let t_coord = uv + texel_size * vec2f(x,  thickness);    // top  coordinate

    return prepass_normal(t_coord) - prepass_normal(d_coord);
}

fn detect_edge_normal(uv: vec2f, thickness: f32) -> f32 {
    let deri_x = abs(
        normal_gradient_x(uv,  thickness, thickness) +
        2.0 * normal_gradient_x(uv,  0.0, thickness) +
        normal_gradient_x(uv, -thickness, thickness));

    let deri_y = abs(
        normal_gradient_y(uv, thickness, thickness) +
        2.0 * normal_gradient_y(uv, 0.0, thickness) +
        normal_gradient_y(uv, -thickness, thickness));

    let x_max = max(deri_x.x, max(deri_x.y, deri_x.z));
    let y_max = max(deri_y.x, max(deri_y.y, deri_y.z));
    
    let grad = max(x_max, y_max);

    return f32(grad > ed_uniform.normal_threshold);
}

// ----------------------
// Color Detection ------
// ----------------------

fn prepass_color(uv: vec2f) -> vec3f {
    return textureSample(screen_texture, texture_sampler, uv).rgb;
}

fn color_gradient_x(uv: vec2f, y: f32, thickness: f32) -> vec3f {
    let l_coord = uv + texel_size * vec2f(-thickness, y);    // left  coordinate
    let r_coord = uv + texel_size * vec2f( thickness, y);    // right coordinate

    return prepass_color(r_coord) - prepass_color(l_coord);
}

fn color_gradient_y(uv: vec2f, x: f32, thickness: f32) -> vec3f {
    let d_coord = uv + texel_size * vec2f(x, -thickness);    // down coordinate
    let t_coord = uv + texel_size * vec2f(x,  thickness);    // top  coordinate

    return prepass_color(t_coord) - prepass_color(d_coord);
}

fn detect_edge_color(uv: vec2f, thickness: f32) -> f32 {
    let deri_x = 
        color_gradient_x(uv,  thickness, thickness) +
        2.0 * color_gradient_x(uv,  0.0, thickness) +
        color_gradient_x(uv, -thickness, thickness);

    let deri_y =
        color_gradient_y(uv,  thickness, thickness) +
        2.0 * color_gradient_y(uv,  0.0, thickness) +
        color_gradient_y(uv, -thickness, thickness);

    let grad = max(length(deri_x), length(deri_y));

    return f32(grad > ed_uniform.color_threshold);
}

var<private> texture_size: vec2f;
var<private> texel_size: vec2f;
var<private> sample_index_i: i32 = 0;

@fragment
fn fragment(
#ifdef MULTISAMPLED
    @builtin(sample_index) sample_index: u32,
#endif
    in: FullscreenVertexOutput
) -> @location(0) vec4f {
#ifdef MULTISAMPLED
    sample_index_i = i32(sample_index);
#endif

    texture_size = vec2f(textureDimensions(screen_texture));
    texel_size = 1.0 / texture_size;

    let uv = in.uv;

    let ndc = vec3f(uv_to_ndc(uv), 1.0);
    let world_position = position_ndc_to_world(ndc);

    let view_direction = calculate_view(world_position);
    let normal = prepass_normal_unpack(uv);
    let fresnel = 1.0 - saturate(dot(normal, view_direction));;

    var edge = 0.0;

#ifdef ENABLE_DEPTH
    let edge_depth = detect_edge_depth(uv, ed_uniform.depth_thickness + 1.2 * noise, fresnel);
    edge = max(edge, edge_depth);
#endif

#ifdef ENABLE_NORMAL
    let edge_normal = detect_edge_normal(uv, ed_uniform.normal_thickness);
    edge = max(edge, edge_normal);
#endif

#ifdef ENABLE_COLOR
    let edge_color = detect_edge_color(uv, ed_uniform.color_thickness);
    edge = max(edge, edge_color);
#endif

    var color = textureSample(screen_texture, texture_sampler, in.uv).rgb;
    color = mix(color, ed_uniform.edge_color.rgb, edge);

    return vec4f(color, 1.0);
}

// MIT License. © Stefan Gustavson, Munrocket
//
fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}