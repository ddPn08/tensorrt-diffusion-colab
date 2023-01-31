# tensorrt-diffusion-colab

[![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/ddPn08/tensorrt-diffusion-colab/blob/main/tensorrt_diffusion.ipynb)

---

A notebook running TensorRT's StableDiffusion demo on Google Colaboratory

## Performance

# Colab T4
512 x 512, 150 Steps
|   Module   |   Latency    |
|------------|--------------|
|    CLIP    |     11.91 ms |
| UNet x 150 |  15094.67 ms |
|    VAE     |    115.69 ms |
|------------|--------------|
|  Pipeline  |  15222.04 ms |

![image](https://user-images.githubusercontent.com/71378929/215680301-fb9bc579-c37d-4bf3-b047-fef66f88c9d2.png)
