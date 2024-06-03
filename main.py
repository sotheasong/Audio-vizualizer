import random
import pygame
import pyaudio
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time


# for pyaudio stuff
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 100

# set the range for the bass and treble freq
bass_range = (20, 400)
treble_range = (10000, 20000)

# initialize the pygame window
pygame.init()
display = (600, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('ohio gyatt')

start_time = time.time()
current_time = time.time()
elapsed_time = current_time - start_time

# set up the OpenGL perspective
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# set up basic OpenGL settings
glEnable(GL_DEPTH_TEST)
glLineWidth(1.5)  # thickness of the wireframe lines
glClearColor(0,0,0,0)  # background color

# for creating the sphere
def create_sphere(r, slices, stacks):
    vertices = []
    for i in range(slices + 1):
        phi = np.pi * i / slices
        for j in range(stacks + 1):
            theta = 2 * np.pi * j / stacks
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            vertices.append((x, y, z))
    return vertices

# for morphing and deforming the vertices of the sphere
def deform_sphere(vertices, deformation_scale_pos, deformation_scale_neg, axis):
    deformed_vertices = []
    for x, y, z in vertices:
        if axis == 'x':
            if x > 0:
                deformation_scale = deformation_scale_pos
            else:
                deformation_scale = deformation_scale_neg
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            phi = np.arctan2(y, x)
            r += deformation_scale * np.sin(5 * theta) * np.cos(5 * phi)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
        deformed_vertices.append((x, y, z))
    return deformed_vertices

def drawSphere(vertices, slices, stacks):
    # change 1 to these 3 for rainbow brr
    color1 = random.random() * (1 + 0.5 * np.sin(elapsed_time))
    color2 = random.random() * (1 + 0.5 * np.sin(elapsed_time))
    color3 = random.random() * (1 + 0.5 * np.sin(elapsed_time))
    glColor3f(color1,color2,color3)  # color of the lines
    glBegin(GL_LINES)
    for i in range(slices):
        for j in range(stacks):
            vertex1 = vertices[i * (stacks + 1) + j]
            vertex2 = vertices[i * (stacks + 1) + (j + 1)]
            vertex3 = vertices[(i + 1) * (stacks + 1) + j]
            vertex4 = vertices[(i + 1) * (stacks + 1) + (j + 1)]

            glVertex3fv(vertex1)
            glVertex3fv(vertex2)
            glVertex3fv(vertex3)

            glVertex3fv(vertex2)
            glVertex3fv(vertex4)
            glVertex3fv(vertex3)
    glEnd()

def main():
    rotation_speed = 0.4  # speed of rotation
    radius = 0.9
    # based on https://www.songho.ca/opengl/gl_sphere.html
    sectors = 23
    stacks = 23

    vertices = create_sphere(radius, sectors, stacks)

    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    running = True
    while running:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Check if audio is silent
        energy = np.sum(np.abs(audio_data)) / len(audio_data)
        if energy < SILENCE_THRESHOLD:
            bass_magnitude = 0
            treble_magnitude = 0
        else:
            # Apply FFT
            fft_result = np.fft.fft(audio_data)

            # Get frequencies corresponding to FFT result
            frequencies = np.fft.fftfreq(len(fft_result), 1.0 / RATE)

            # Find indices corresponding to frequency ranges
            bass_indices = np.where(np.logical_and(frequencies >= bass_range[0], frequencies <= bass_range[1]))[0]
            treble_indices = np.where(np.logical_and(frequencies >= treble_range[0], frequencies <= treble_range[1]))[0]

            # Calculate magnitude for each frequency range
            bass_magnitude = np.mean(np.abs(fft_result[bass_indices]))
            treble_magnitude = np.mean(np.abs(fft_result[treble_indices]))

        # Dynamically update the deformation scale based on time
        deformation_scale_pos = bass_magnitude/(10000000*0.3) * (1 + 0.5 * np.sin(elapsed_time))
        deformation_scale_neg = treble_magnitude/(100000 * 1.2) * (1 + 0.5 * np.sin(elapsed_time + np.pi))

        # Deform each half of the sphere separately
        deformed_vertices = deform_sphere(vertices, deformation_scale_pos, deformation_scale_neg, axis="x")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Rotate the sphere
        glRotatef(rotation_speed, 1, 1, 1)

        # Draw the solid sphere for each half
        drawSphere(deformed_vertices, sectors, stacks)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()
