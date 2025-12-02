from PIL import Image, ImageDraw

# =======================
# CONFIGURATION
# =======================

# Physical dimensions
square_size_mm = 10.0       # size of one square in millimetres
num_cols       = 7         # number of squares horizontally
num_rows       = 5         # number of squares vertically

# Printing / image resolution
dpi = 600                  # dots per inch (600 is good for printing)

# Output filenames
png_filename = f"checkerboard_{int(square_size_mm)}mm_{num_cols}x{num_rows}_{dpi}dpi.png"
pdf_filename = f"checkerboard_{int(square_size_mm)}mm_{num_cols}x{num_rows}_{dpi}dpi.pdf"

# =======================
# CALCULATIONS
# =======================

# Convert mm → inches → pixels
mm_per_inch = 25.4
square_size_inch = square_size_mm / mm_per_inch
square_size_px = int(round(square_size_inch * dpi))

img_width_px = num_cols * square_size_px
img_height_px = num_rows * square_size_px

print(f"Square size (px): {square_size_px}")
print(f"Image size (px): {img_width_px} x {img_height_px}")
print(f"Physical size (mm): {num_cols * square_size_mm} x {num_rows * square_size_mm}")

# =======================
# CREATE CHECKERBOARD
# =======================

# "L" = 8-bit grayscale. Start with white background (255).
img = Image.new("L", (img_width_px, img_height_px), 255)
draw = ImageDraw.Draw(img)

for row in range(num_rows):
    for col in range(num_cols):
        # Alternate black / white
        if (row + col) % 2 == 0:
            x0 = col * square_size_px
            y0 = row * square_size_px
            x1 = x0 + square_size_px
            y1 = y0 + square_size_px
            draw.rectangle([x0, y0, x1, y1], fill=0)  # black square

# =======================
# SAVE IMAGES
# =======================

# Save as PNG
img.save(png_filename, dpi=(dpi, dpi))
print(f"Saved PNG: {png_filename}")

# Optionally: save as PDF (comment this out if you don't need PDF)
img.save(pdf_filename, "PDF", resolution=dpi)
print(f"Saved PDF: {pdf_filename}")
