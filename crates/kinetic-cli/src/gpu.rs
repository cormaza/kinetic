//! Which GPU a model runs on: CUDA first, then Vulkan, else the CPU.

use kinetic_models::LoadError;

use crate::{Device, Result};

/// The GPU kinds in the order `auto` tries them, with their names.
const AUTO_ORDER: [(Device, &str); 2] = [(Device::Cuda, "CUDA"), (Device::Vulkan(None), "Vulkan")];

/// The first backend that `start` makes for a GPU kind `device` allows,
/// and its kind; `None` when the model should run on the CPU. `start`
/// returns `None` for kinds this build lacks. A GPU asked for by name must
/// work; under `auto`, a GPU that does not is skipped with one line saying
/// why.
pub fn pick<T>(
    device: Device,
    mut start: impl FnMut(Device) -> Option<std::result::Result<T, LoadError>>,
) -> Result<Option<(T, Device)>> {
    let explicit;
    let order: &[(Device, &str)] = match device {
        Device::Cpu => return Ok(None),
        Device::Auto => &AUTO_ORDER,
        Device::Cuda => {
            explicit = [(Device::Cuda, "CUDA")];
            &explicit
        }
        Device::Vulkan(i) => {
            explicit = [(Device::Vulkan(i), "Vulkan")];
            &explicit
        }
    };
    let mut skipped = Vec::new();
    for &(kind, name) in order {
        match start(kind) {
            Some(Ok(backend)) => {
                if !skipped.is_empty() {
                    eprintln!("kinetic: running through {name}; {}", skipped.join("; "));
                }
                return Ok(Some((backend, kind)));
            }
            Some(Err(e)) if device != Device::Auto => return Err(e.into()),
            None if device != Device::Auto => {
                return Err(format!("this kinetic was built without {name}").into())
            }
            Some(Err(e)) => skipped.push(format!("{name} is not usable: {e}")),
            None => {}
        }
    }
    if !skipped.is_empty() {
        eprintln!("kinetic: running on the CPU; {}", skipped.join("; "));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parses_vulkan_indices_and_aliases() {
        assert_eq!("vulkan".parse::<Device>().unwrap(), Device::Vulkan(None));
        assert_eq!(
            "vulkan0".parse::<Device>().unwrap(),
            Device::Vulkan(Some(0))
        );
        assert_eq!(
            "vulkan1".parse::<Device>().unwrap(),
            Device::Vulkan(Some(1))
        );
        assert_eq!(
            "VULKAN2".parse::<Device>().unwrap(),
            Device::Vulkan(Some(2))
        );
        assert_eq!("cuda0".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("none".parse::<Device>().unwrap(), Device::Cpu);
        assert!("vulkanX".parse::<Device>().is_err());
    }
}
