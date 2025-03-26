import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog
from matplotlib.widgets import Button

class FrequencyDomainFilter:
    def __init__(self):
        self.selected_points = []
        self.current_image_path = None
        self.current_image = None
        self.f_shift = None
        self.magnitude_spectrum = None
        self.filter_radius = 10
        self.base_path = None
        self.input_folder = None
        self.output_folder = None
        self.all_filter_params = {}
        
        self.setup_directories()
    
    def setup_directories(self):
        root = tk.Tk()
        root.withdraw()
        
        self.base_path = filedialog.askdirectory(title="Select Project Base Directory")
        
        if not self.base_path:
            self.base_path = os.getcwd()
        
        self.base_path = Path(self.base_path)
        self.input_folder = self.base_path / "Input"
        self.output_folder = self.base_path / "Output"
        
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
    
    def select_images(self):
        root = tk.Tk()
        root.withdraw()
        
        file_paths = filedialog.askopenfilenames(
            initialdir=self.input_folder,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        )
        
        return list(file_paths)
    
    def process_image(self, image_path):
        self.current_image_path = image_path
        filename = os.path.basename(image_path)
        
        self.current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        f = np.fft.fft2(self.current_image)
        self.f_shift = np.fft.fftshift(f)
        
        self.magnitude_spectrum = 20 * np.log(np.abs(self.f_shift) + 1)
        
        self.selected_points = []
        
        self.show_interactive_spectrum()
        
        self.filter_radius = simpledialog.askinteger(
            "Filter Radius",
            f"Enter the filter radius for {filename}:",
            initialvalue=10,
            minvalue=1,
            maxvalue=100
        )
        
        if self.filter_radius is None:
            self.filter_radius = 10
        
        self.all_filter_params[filename] = {
            "center_points": self.selected_points,
            "filter_radius": self.filter_radius
        }
        
        self.apply_filter_and_save(filename)
    
    def show_interactive_spectrum(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.2)
        
        ax[0].imshow(self.current_image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        spectrum_img = ax[1].imshow(self.magnitude_spectrum, cmap='viridis')
        ax[1].set_title('Magnitude Spectrum (log scale)\nClick to select noise points')
        ax[1].axis('off')
        
        plt.colorbar(spectrum_img, ax=ax[1], shrink=0.7)
        
        points_text = plt.figtext(0.5, 0.05, "Selected Points: []", ha="center", fontsize=10)
        
        def onclick(event):
            if event.inaxes == ax[1]:
                ix, iy = int(event.xdata), int(event.ydata)
                self.selected_points.append((ix, iy))
                
                points_text.set_text(f"Selected Points: {self.selected_points}")
                
                ax[1].plot(ix, iy, 'ro', markersize=5)
                fig.canvas.draw()
        
        def clear_points(event):
            self.selected_points = []
            points_text.set_text("Selected Points: []")
            
            ax[1].clear()
            ax[1].imshow(self.magnitude_spectrum, cmap='viridis')
            ax[1].set_title('Magnitude Spectrum (log scale)\nClick to select noise points')
            ax[1].axis('off')
            fig.canvas.draw()
        
        def preview_filter(event):
            if not self.selected_points:
                return
            
            filter_radius = simpledialog.askinteger(
                "Preview Filter",
                "Enter filter radius for preview:",
                initialvalue=10,
                minvalue=1,
                maxvalue=100
            )
            
            if filter_radius is None:
                return
                
            filtered_f_shift = self.apply_notch_filter(self.f_shift.copy(), self.selected_points, filter_radius)
            filtered_magnitude = 20 * np.log(np.abs(filtered_f_shift) + 1)
            
            f_ishift = np.fft.ifftshift(filtered_f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            img_back = np.uint8(img_back)
            
            preview_fig, preview_ax = plt.subplots(1, 3, figsize=(18, 6))
            preview_ax[0].imshow(self.current_image, cmap='gray')
            preview_ax[0].set_title('Original Image')
            preview_ax[0].axis('off')
            
            preview_ax[1].imshow(filtered_magnitude, cmap='viridis')
            preview_ax[1].set_title('Filtered Spectrum')
            preview_ax[1].axis('off')
            
            preview_ax[2].imshow(img_back, cmap='gray')
            preview_ax[2].set_title('Filtered Image')
            preview_ax[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        def add_symmetric(event):
            if not self.selected_points:
                return
            
            rows, cols = self.current_image.shape
            center_y, center_x = rows // 2, cols // 2
            
            current_points = self.selected_points.copy()
            
            for point in current_points:
                x, y = point
                sym_x = 2 * center_x - x
                sym_y = 2 * center_y - y
                
                if (sym_x, sym_y) not in self.selected_points:
                    self.selected_points.append((sym_x, sym_y))
                    ax[1].plot(sym_x, sym_y, 'ro', markersize=5)
            
            points_text.set_text(f"Selected Points: {self.selected_points}")
            fig.canvas.draw()
        
        def done_selecting(event):
            plt.close(fig)
        
        clear_ax = plt.axes([0.15, 0.05, 0.1, 0.04])
        clear_button = Button(clear_ax, 'Clear Points')
        clear_button.on_clicked(clear_points)
        
        preview_ax = plt.axes([0.3, 0.05, 0.1, 0.04])
        preview_button = Button(preview_ax, 'Preview')
        preview_button.on_clicked(preview_filter)
        
        symmetric_ax = plt.axes([0.45, 0.05, 0.15, 0.04])
        symmetric_button = Button(symmetric_ax, 'Add Symmetric')
        symmetric_button.on_clicked(add_symmetric)
        
        done_ax = plt.axes([0.65, 0.05, 0.1, 0.04])
        done_button = Button(done_ax, 'Done')
        done_button.on_clicked(done_selecting)
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.show()
    
    def apply_notch_filter(self, f_shift, center_points, filter_radius):
        rows, cols = f_shift.shape
        mask = np.ones((rows, cols), np.float32)
        
        y, x = np.ogrid[:rows, :cols]
        
        for point in center_points:
            px, py = point
            dist_from_center = np.sqrt((x - px)**2 + (y - py)**2)
            mask[dist_from_center <= filter_radius] = 0
            
        return f_shift * mask
    
    def apply_filter_and_save(self, filename):
        if not self.selected_points:
            return
        
        center_points = self.selected_points
        filter_radius = self.filter_radius
        
        filtered_f_shift = self.apply_notch_filter(self.f_shift, center_points, filter_radius)
        
        filtered_magnitude = 20 * np.log(np.abs(filtered_f_shift) + 1)
        
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
        
        output_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_path, img_back)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231), plt.imshow(self.current_image, cmap='gray')
        plt.title('Original Image'), plt.axis('off')
        
        plt.subplot(232), plt.imshow(self.magnitude_spectrum, cmap='viridis')
        plt.title('Original Spectrum'), plt.axis('off')
        
        plt.subplot(233), plt.imshow(filtered_magnitude, cmap='viridis')
        plt.title('Filtered Spectrum'), plt.axis('off')
        
        plt.subplot(234), plt.imshow(img_back, cmap='gray')
        plt.title('Filtered Image'), plt.axis('off')
        
        param_text = f"Filter parameters:\n"
        param_text += f"Points: {center_points}\n"
        param_text += f"Radius: {filter_radius}"
        plt.figtext(0.7, 0.3, param_text, wrap=True, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"results_{filename}"))
        plt.show()
    
    def process_all_images(self):
        image_paths = self.select_images()
        
        if not image_paths:
            return
        
        for image_path in image_paths:
            self.process_image(image_path)
        
        self.save_parameters()
    
    def save_parameters(self):
        param_file = os.path.join(self.output_folder, "filter_parameters.txt")
        
        with open(param_file, 'w') as f:
            f.write("Frequency Domain Filter Parameters\n")
            f.write("=================================\n\n")
            
            for filename, params in self.all_filter_params.items():
                f.write(f"Image: {filename}\n")
                f.write(f"Points: {params['center_points']}\n")
                f.write(f"Radius: {params['filter_radius']}\n")
                f.write("---------------------------------\n\n")

if __name__ == "__main__":
    filter_app = FrequencyDomainFilter()
    filter_app.process_all_images()