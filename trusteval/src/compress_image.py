from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def compress_image(input_path, output_path, quality=85, max_size_kb=100, max_attempts=20):
    """
    Compress any image to JPG format with specified quality and size limit.

    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed JPG image.
    :param quality: Initial quality of the compressed image (1-95). Default is 85.
    :param max_size_kb: Maximum size of the compressed image in KB. Default is 100 KB.
    :param max_attempts: Maximum number of compression attempts. Default is 10.
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB mode if necessary
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Ensure the output path has a .jpg extension
            output_path = os.path.splitext(output_path)[0] + '.jpg'

            for attempt in range(max_attempts):
                img.save(output_path, "JPEG", quality=quality)

                if os.path.getsize(output_path) <= max_size_kb * 1024:
                    print(f"Image compressed successfully to JPG in {attempt + 1} attempts: {output_path}")
                    break

                quality -= 5
                if quality < 20:
                    print(f"Cannot compress {input_path} to desired size. Saving with minimum quality.")
                    img.save(output_path, "JPEG", quality=20)
                    break
            else:
                print(f"Image could not be compressed to {max_size_kb}KB after {max_attempts} attempts: {output_path}")

    except IOError:
        print(f"Cannot open image file: {input_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def compress_images_in_folder(folder_path, output_folder, quality=85, max_size_kb=100, max_workers=4):
    """
    Compress all images in a folder using multithreading and save them in the specified output folder.

    :param folder_path: Path to the folder containing images to compress.
    :param output_folder: Path to the folder to save compressed images.
    :param quality: Initial quality of the compressed images. Default is 85.
    :param max_size_kb: Maximum size of the compressed images in KB. Default is 100 KB.
    :param max_workers: Number of threads to use for concurrent compression. Default is 4.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect all images from the folder
    images_to_compress = [
        (os.path.join(folder_path, filename), os.path.join(output_folder, filename))
        for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    # Use ThreadPoolExecutor for multithreaded compression
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compress_image, input_path, output_path, quality, max_size_kb)
            for input_path, output_path in images_to_compress
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An image compression task generated an exception: {exc}")

compress_image('/Users/admin/Documents/GitHub/TrustGen.github.io/assets/img/background/37.jpg', '/Users/admin/Documents/GitHub/TrustGen.github.io/assets/img/background/37.jpg', quality=85, max_size_kb=100)
