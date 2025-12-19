# Guide: Adding Images and Videos to GitHub Pages

## Adding Images

### Option 1: Store Images in Repository (Recommended)

1. Create an `images` folder in the `docs/` directory:
   ```bash
   mkdir docs/images
   ```

2. Add your image files to `docs/images/`:
   ```bash
   cp your-image.jpg docs/images/
   ```

3. Reference in HTML:
   ```html
   <img src="images/your-image.jpg" alt="Description">
   ```

### Option 2: Use External URLs

If your images are hosted elsewhere (Imgur, Google Photos, etc.):
```html
<img src="https://example.com/path/to/image.jpg" alt="Description">
```

## Adding YouTube Videos

1. Get your YouTube video ID:
   - URL format: `https://www.youtube.com/watch?v=VIDEO_ID`
   - The `VIDEO_ID` is the part after `v=`

2. Replace `VIDEO_ID` in the HTML:
   ```html
   <div class="video-container">
       <iframe 
           src="https://www.youtube.com/embed/YOUR_VIDEO_ID" 
           allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
           allowfullscreen>
       </iframe>
   </div>
   ```

**Example:**
- Video URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- Use: `https://www.youtube.com/embed/dQw4w9WgXcQ`

## Adding Google Drive Videos

1. Upload your video to Google Drive

2. Right-click the video → **Get link** → Set to **"Anyone with the link can view"**

3. Get the File ID from the sharing link:
   - Link format: `https://drive.google.com/file/d/FILE_ID/view`
   - Copy the `FILE_ID` (the long string between `/d/` and `/view`)

4. Use this format in HTML:
   ```html
   <div class="video-container">
       <iframe 
           src="https://drive.google.com/file/d/YOUR_FILE_ID/preview" 
           allow="autoplay" 
           allowfullscreen>
       </iframe>
   </div>
   ```

**Important:** Make sure the video sharing settings allow "Anyone with the link can view"

## Quick Reference

### Single Image
```html
<div class="image-container">
    <img src="images/filename.jpg" alt="Description">
    <p class="image-caption">Caption text here</p>
</div>
```

### Image Grid (Multiple Images)
```html
<div class="image-grid">
    <div class="image-container">
        <img src="images/image1.jpg" alt="Description 1">
        <p class="image-caption">Caption 1</p>
    </div>
    <div class="image-container">
        <img src="images/image2.jpg" alt="Description 2">
        <p class="image-caption">Caption 2</p>
    </div>
</div>
```

### Single Video
```html
<div class="video-container">
    <iframe 
        src="https://www.youtube.com/embed/VIDEO_ID" 
        allowfullscreen>
    </iframe>
</div>
<p class="image-caption">Video description</p>
```

### Video Grid (Multiple Videos)
```html
<div class="video-grid">
    <div>
        <div class="video-container">
            <iframe src="https://www.youtube.com/embed/VIDEO_ID_1" allowfullscreen></iframe>
        </div>
        <p class="image-caption">Video 1 description</p>
    </div>
    <div>
        <div class="video-container">
            <iframe src="https://drive.google.com/file/d/FILE_ID/preview" allowfullscreen></iframe>
        </div>
        <p class="image-caption">Video 2 description</p>
    </div>
</div>
```

## File Size Recommendations

- **Images:** Keep under 2MB each for faster loading. Use JPEG for photos, PNG for screenshots.
- **Videos:** Use YouTube or Google Drive (no size limit), or compress videos if hosting directly.

## Testing

After adding media:
1. Commit and push changes
2. Wait for GitHub Pages to rebuild
3. Visit your site and verify images/videos load correctly
