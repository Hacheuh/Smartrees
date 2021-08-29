from PIL import Image
import glob

def create_gif(pathname):
    '''This function transforms pngs into a gif, and saves it    
    '''
    frames = []
    imgs = glob.glob(pathname+"/*_NDVI.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    print(frames)
    # Save into a GIF file that loops forever
    frames[0].save('png_to_gif.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=300, loop=0)
    return None