from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap
import os
from typing import Optional

app = FastAPI(title="Text to Handwriting API", version="1.0.0")

class TextRequest(BaseModel):
    text: str
    font_size: Optional[int] = 32
    ink_color: Optional[str] = "#000055"
    line_spacing: Optional[int] = 40
    margin: Optional[int] = 50

class HandwritingGenerator:
    def __init__(self):
        self.font_path = self._get_font_path()
    
    def _get_font_path(self):
        """Get the path to a handwriting-style font suitable for English text"""
        # Try to find English handwriting fonts first
        # Avoid Hindi_Type.ttf as it renders English text in Hindi script
        font_paths = [
            # Downloaded English handwriting fonts
            os.path.join("fonts", "Caveat-Regular.ttf"),
            # Common Windows handwriting fonts
            "C:/Windows/Fonts/segoepr.ttf",  # Segoe Print - best handwriting font on Windows
            "C:/Windows/Fonts/bradley.ttf",  # Bradley Hand ITC
            "C:/Windows/Fonts/comic.ttf",    # Comic Sans MS
            "C:/Windows/Fonts/comicbd.ttf",  # Comic Sans MS Bold
            # Standard fonts as fallback
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            # Last resort - avoid Hindi font for English text
            # os.path.join("fonts", "Hindi_Type.ttf"),  # Commented out
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                print(f"Using font: {font_path}")  # Debug info
                return font_path
        
        # If no custom font found, use default
        print("No custom font found, using default")  # Debug info
        return None
    
    def _hex_to_rgb(self, hex_color: str):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _get_font(self, size: int):
        """Load font with specified size"""
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size)
            else:
                # Try to use system fonts that render English properly
                system_fonts = [
                    # Windows handwriting-style fonts
                    "C:/Windows/Fonts/segoepr.ttf",  # Segoe Print
                    "C:/Windows/Fonts/comic.ttf",    # Comic Sans MS
                    "C:/Windows/Fonts/comicbd.ttf",  # Comic Sans MS Bold
                    # Standard fonts
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf",
                    "C:/Windows/Fonts/times.ttf",
                    # Simple font names (system will resolve)
                    "arial.ttf",
                    "calibri.ttf",
                    "times.ttf"
                ]
                for font_name in system_fonts:
                    try:
                        return ImageFont.truetype(font_name, size)
                    except:
                        continue
                # If all else fails, use default font
                return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
    
    def generate_image(self, text: str, font_size: int = 32, ink_color: str = "#000055", 
                      line_spacing: int = 40, margin: int = 50) -> Image.Image:
        """Generate handwriting-style image from text"""
        
        # Convert hex color to RGB
        rgb_color = self._hex_to_rgb(ink_color)
        
        # Load font
        font = self._get_font(font_size)
        
        # Split text into lines and wrap long lines
        lines = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                wrapped_lines = textwrap.wrap(paragraph, width=60)  # Adjust width as needed
                lines.extend(wrapped_lines)
            else:
                lines.append('')  # Preserve empty lines
        
        # Calculate image dimensions
        max_width = 0
        line_heights = []
        
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1, 1), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        
        for line in lines:
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            max_width = max(max_width, line_width)
            line_heights.append(line_height)
        
        # Calculate total image size
        total_height = sum(line_heights) + (len(lines) - 1) * (line_spacing - max(line_heights)) + 2 * margin
        total_width = max_width + 2 * margin
        
        # Ensure minimum size
        total_width = max(total_width, 400)
        total_height = max(total_height, 200)
        
        # Create final image with white background
        img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y_position = margin
        for i, line in enumerate(lines):
            if line.strip():  # Only draw non-empty lines
                draw.text((margin, y_position), line, fill=rgb_color, font=font)
            y_position += line_heights[i] + (line_spacing - line_heights[i]) if i < len(line_heights) else 0
        
        return img

# Initialize the generator
generator = HandwritingGenerator()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Text to Handwriting API is running"}

@app.post("/generate-handwriting")
async def generate_handwriting(request: TextRequest):
    """
    Generate handwriting-style image from text
    
    Args:
        request: TextRequest containing text and optional styling parameters
    
    Returns:
        StreamingResponse: PNG image of the handwritten text
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate the image
        img = generator.generate_image(
            text=request.text,
            font_size=request.font_size,
            ink_color=request.ink_color,
            line_spacing=request.line_spacing,
            margin=request.margin
        )
        
        # Convert image to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=handwriting.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)