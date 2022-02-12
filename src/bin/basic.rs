use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::Version;

// Create physical devices
use vulkano::device::physical::PhysicalDevice;

// For devices
use vulkano::device::{Device, DeviceExtensions, Features};

// For accessable buffer
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};

// For command buffer
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};

// Submit commands
use vulkano::sync;
use vulkano::sync::GpuFuture;

// Pipline 
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

fn main()
{
    // let required_extensions = vulkano_win::required_extensions();
    let instance =  Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None)
        .expect("failed to create an instance");

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    // queue for cpu operations
    // for family in physical.queue_families() {
    //     println!("Found a queue family with {:?} queue(s)", family.queues_count());
    // }
    let queue_families = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("coundn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(physical, &Features::none(), &DeviceExtensions::none(), [(queue_families, 0.5)].iter().cloned())
            .expect("Failed to create device")
    };

    let queue = queues.next().unwrap();

    enum PracticeBufferType
    {
        Simple,
        Compute,
    }
    let buffer_type = PracticeBufferType::Compute;

    match buffer_type
    {
        PracticeBufferType::Simple => {
            // Simple example
            // accesable buffer for store and operate datas
            let source_content = 0..64; 
            let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, source_content)
                .expect("failed to create buffer");

            let destination_content = (0..64).map(|_| 0);
            let destination = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, destination_content)
                .expect("failed to create buffer");

            // Command buffer
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(), 
                CommandBufferUsage::OneTimeSubmit,
            ).unwrap();

            builder.copy_buffer(source.clone(), destination.clone()).unwrap();

            let command_buffer = builder.build().unwrap();

            // Subimit and try get data when GPU finished operation
            let future = sync::now(device.clone())
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush() // signal to the cpu and start executing
                .unwrap();

            // wait for GPU
            future.wait(None).unwrap();

            // Read data from destination
            let src_content = source.read().unwrap();
            let dest_content = destination.read().unwrap();
            assert_eq!(&*src_content, &*dest_content);
            println! ("Simple buffer ok!")
        }

        PracticeBufferType::Compute => {
            // Compute example
            let data_iter = 0..65535;
            let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
                .expect("failed to create buffer");

            mod cs {
                vulkano_shaders::shader!{
                    ty: "compute",
                    src: "
                        #version 450

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        
                        layout(set = 0, binding = 0) buffer Data {
                            uint data[];
                        } buf;

                        void main() {
                            uint idx = gl_GlobalInvocationID.x;
                            buf.data[idx] *= 12;
                        }
                    "
                }
            }

            let shader = cs::load(device.clone())
                .expect("Failed to create shader module!");

            let compute_pipeline = ComputePipeline::new(
                device.clone(), 
                shader.entry_point("main").unwrap(), 
                &(), 
                None, 
                |_| (),
            ).expect("Failed to create compute pipeline");

            let layout = compute_pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .unwrap();

            let set = PersistentDescriptorSet::new(
                layout.clone(), 
                [WriteDescriptorSet::buffer(0, data_buffer.clone())],
                ).unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(), 
                    CommandBufferUsage::OneTimeSubmit,
                ).unwrap();

            builder.bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute, 
                    compute_pipeline.layout().clone(), 
                    0, 
                    set)
                .dispatch([1024, 1, 1])
                .unwrap();

            let command_buffer = builder.build().unwrap();

            let future = sync::now(device.clone())
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();

            future.wait(None).unwrap();

            let content = data_buffer.read().unwrap();

            for (n, val) in content.iter().enumerate() {
                assert_eq!(*val, n as u32 * 12);
            }
            
            println!("Compute shader succeeded!");            
        }
    }

}