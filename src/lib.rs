use bevy::{
    asset::load_internal_asset,
    core_pipeline::{
        core_3d::{
            graph::{Core3d, Node3d},
            DEPTH_TEXTURE_SAMPLING_SUPPORTED,
        },
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, UniformComponentPlugin,
        },
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        sync_component::SyncComponentPlugin,
        sync_world::RenderEntity,
        view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, RenderApp,
    },
};
use binding_types::texture_depth_2d;

const EDGE_DETECTION_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(98765432109876543210987654321098765);

/// An edge detection post-processing plugin based on the sobel filter.
pub struct EdgeDetectionPlugin {
    pub after: Node3d,
    pub before: Node3d,
}

impl Default for EdgeDetectionPlugin {
    fn default() -> Self {
        Self {
            after: Node3d::Tonemapping,
            before: Node3d::Fxaa,
        }
    }
}

impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            EDGE_DETECTION_SHADER_HANDLE,
            "edge_detection.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<EdgeDetection>();

        app.add_plugins(SyncComponentPlugin::<EdgeDetection>::default())
            .add_plugins(UniformComponentPlugin::<EdgeDetectionUniform>::default());

        // We need to get the render app from the main app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                ExtractSchedule,
                EdgeDetectionUniform::extract_edge_detection_settings,
            )
            .add_render_graph_node::<ViewNodeRunner<EdgeDetectionNode>>(Core3d, EdgeDetectionLabel)
            .add_render_graph_edges(
                Core3d,
                (self.after.clone(), EdgeDetectionLabel, self.before.clone()),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<EdgeDetectionPipeline>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct EdgeDetectionLabel;

// The post process node used for the render graph
#[derive(Default)]
struct EdgeDetectionNode;

impl ViewNode for EdgeDetectionNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
        &'static DynamicUniformIndex<EdgeDetectionUniform>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, prepass_textures, view_uniform_index, ed_uniform_index): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let edge_detection_pipeline = world.resource::<EdgeDetectionPipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(edge_detection_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let (Some(depth_texture), Some(normal_texture)) =
            (&prepass_textures.depth, &prepass_textures.normal)
        else {
            return Ok(());
        };

        let Some(view_uniforms_binding) = world.resource::<ViewUniforms>().uniforms.binding()
        else {
            return Ok(());
        };

        let Some(ed_uniform_binding) = world
            .resource::<ComponentUniforms<EdgeDetectionUniform>>()
            .uniforms()
            .binding()
        else {
            return Ok(());
        };

        // This will start a new "post process write", obtaining two texture
        // views from the view target - a `source` and a `destination`.
        // `source` is the "current" main texture and you _must_ write into
        // `destination` because calling `post_process_write()` on the
        // [`ViewTarget`] will internally flip the [`ViewTarget`]'s main
        // texture to the `destination` texture. Failing to do so will cause
        // the current main texture information to be lost.
        let post_process = view_target.post_process_write();

        // The bind_group gets created each frame.
        //
        // Normally, you would create a bind_group in the Queue set,
        // but this doesn't work with the post_process_write().
        // The reason it doesn't work is because each post_process_write will alternate the source/destination.
        // The only way to have the correct source/destination for the bind_group
        // is to make sure you get it during the node execution.
        let bind_group = render_context.render_device().create_bind_group(
            "edge_detection_bind_group",
            &edge_detection_pipeline.layout,
            // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
            &BindGroupEntries::sequential((
                // Make sure to use the source view
                post_process.source,
                // Use the sampler created for the pipeline
                &edge_detection_pipeline.sampler,
                // Use depth prepass
                &depth_texture.texture.default_view,
                // Use normal prepass
                &normal_texture.texture.default_view,
                // view uniform binding
                view_uniforms_binding,
                // Set the uniform binding
                ed_uniform_binding,
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("edge_detection_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &bind_group,
            &[view_uniform_index.offset, ed_uniform_index.index()],
        );
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

// This contains global data used by the render pipeline. This will be created once on startup.
#[derive(Resource)]
struct EdgeDetectionPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for EdgeDetectionPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // We need to define the bind group layout used for our pipeline
        let layout = render_device.create_bind_group_layout(
            "edge_detection_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                // The layout entries will only be visible in the fragment stage
                ShaderStages::FRAGMENT,
                (
                    // The screen texture
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // The sampler that will be used to sample the screen texture
                    sampler(SamplerBindingType::Filtering),
                    // depth prepass
                    texture_depth_2d(),
                    // normal prepass
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // view
                    uniform_buffer::<ViewUniform>(true),
                    // The uniform that will control the effect
                    uniform_buffer::<EdgeDetectionUniform>(true),
                ),
            ),
        );

        // We can create the sampler here since it won't change at runtime and doesn't depend on the view
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            // This will add the pipeline to the cache and queue its creation
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("edge_detection_pipeline".into()),
                layout: vec![layout.clone()],
                // This will setup a fullscreen triangle for the vertex state
                vertex: fullscreen_shader_vertex_state(),
                fragment: Some(FragmentState {
                    shader: EDGE_DETECTION_SHADER_HANDLE,
                    shader_defs: vec![],
                    // Make sure this matches the entry point of your shader.
                    // It can be anything as long as it matches here and in the shader.
                    entry_point: "fragment".into(),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::bevy_default(),
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                // All of the following properties are not important for this effect so just use the default values.
                // This struct doesn't have the Default trait implemented because not all fields can have a default value.
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component, Default)]
pub struct EdgeDetection {
    /// Depth threshold, used to detect edges with significant depth changes.
    /// Areas where the depth variation exceeds this threshold will be marked as edges.
    pub depth_threshold: f32,
    /// Normal threshold, used to detect edges with significant normal direction changes.
    /// Areas where the normal direction variation exceeds this threshold will be marked as edges.
    pub normal_threshold: f32,
    /// Color threshold, used to detect edges with significant color changes.
    /// Areas where the color variation exceeds this threshold will be marked as edges.
    pub color_threshold: f32,

    /// Steep angle threshold, used to adjust the depth threshold when viewing surfaces at steep angles.
    /// When the angle between the view direction and the surface normal is very steep, the depth gradient
    /// can appear artificially large, causing non-edge regions to be mistakenly detected as edges.
    /// This threshold defines the angle at which the depth threshold adjustment begins to take effect.
    ///
    /// Range: [0.0, 1.0]
    pub steep_angle_threshold: f32,

    /// Edge color, used to draw the detected edges.
    /// Typically a high-contrast color (e.g., red or black) to visually highlight the edges.
    pub edge_color: Color,
}

impl Default for EdgeDetection {
    fn default() -> Self {
        Self {
            depth_threshold: 1.0,
            normal_threshold: 0.8,
            color_threshold: 0.0,

            edge_color: Color::BLACK.into(),

            steep_angle_threshold: 0.5,
        }
    }
}

#[derive(Component, Clone, Copy, ShaderType, ExtractComponent)]
struct EdgeDetectionUniform {
    depth_threshold: f32,
    normal_threshold: f32,
    color_threshold: f32,
    steep_angle_threshold: f32,
    edge_color: LinearRgba,
}

impl EdgeDetectionUniform {
    fn extract_edge_detection_settings(
        mut commands: Commands,
        mut query: Extract<Query<(RenderEntity, &EdgeDetection)>>,
    ) {
        if !DEPTH_TEXTURE_SAMPLING_SUPPORTED {
            info_once!(
                "Disable edge detection on this platform because depth textures aren't supported correctly"
            );
            return;
        }

        for (entity, edge_detection) in query.iter_mut() {
            let mut entity_commands = commands
                .get_entity(entity)
                .expect("Edge Detection entity wasn't synced.");

            entity_commands.insert(EdgeDetectionUniform::from(edge_detection));
        }
    }
}

impl From<&EdgeDetection> for EdgeDetectionUniform {
    fn from(ed: &EdgeDetection) -> Self {
        Self {
            depth_threshold: ed.depth_threshold,
            normal_threshold: ed.normal_threshold,
            color_threshold: ed.color_threshold,
            steep_angle_threshold: ed.steep_angle_threshold,
            edge_color: ed.edge_color.into(),
        }
    }
}
