"""Exit non-zero unless SAPIEN can create a Vulkan renderer in this container."""
import sapien
sapien.SapienRenderer()
print("render ok")
