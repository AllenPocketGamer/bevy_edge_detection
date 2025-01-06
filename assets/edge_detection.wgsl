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

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = vec2u(in.position.xy);

    let color = textureSample(screen_texture, texture_sampler, in.uv).rgb;
    let normal = textureLoad(normal_prepass_texture, frag_coord, 0).rgb;
    let depth = textureLoad(depth_prepass_texture, frag_coord, 0);
    // let color = textureLoad(screen_texture, frag_coord, 0).rgb;
    return vec4f(color, 1.0);
}

