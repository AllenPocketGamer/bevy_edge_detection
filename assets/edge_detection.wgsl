#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(3) var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> settings: EdgeDetectionUniform;

struct EdgeDetectionUniform {
    intensity: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Chromatic aberration strength
    let offset_strength = settings.intensity;

    let frag_coord = vec2u(in.position.xy);

    let color = textureSample(screen_texture, texture_sampler, in.uv).rgb;
    let normal = textureLoad(normal_prepass_texture, frag_coord, 0).rgb;
    let depth = textureLoad(depth_prepass_texture, frag_coord, 0);
    // let color = textureLoad(screen_texture, frag_coord, 0).rgb;
    return vec4f(normal, 1.0);
}

