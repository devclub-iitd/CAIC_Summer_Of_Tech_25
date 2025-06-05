# Steganography - The Art of Hidden Messages

Steganography hides secret information inside ordinary-looking files. Unlike cryptography (which scrambles data), steganography hides the fact that secret data exists at all!

---

## Image Steganography

### Detection Techniques

**Visual Inspection:**
- Examine the image carefully for unusual patterns
- Check if parts look "noisy" or slightly different
- Look for text or patterns in solid color areas

**File Size Analysis:**
- Compare file size to what you'd expect for that image quality
- A simple 100x100 pixel image that's 2MB might contain hidden data

**Metadata Examination:**
Images contain hidden metadata (EXIF data) that might contain secrets:
- GPS coordinates, camera settings, device information
- Custom fields where data can be hidden

### Common Steganography Methods

**LSB (Least Significant Bit) Hiding:**
The most common image steganography technique.

**How it works:**
```
1. Take a pixel with RGB values: (255, 128, 64)
2. In binary: (11111111, 10000000, 01000000)
3. Replace the last bit of each color with secret data bits
4. Result: (11111110, 10000001, 01000001) = (254, 129, 65)
5. Human eye can't detect this tiny color change!
```

**Steghide Method:**
- Password-protected hiding in JPEG and BMP images
- Changes pixel values slightly to embed data
- Requires password to extract hidden data

### Analysis Tools

**Online Tools:**
- **StegOnline**: https://georgeom.net/StegOnline/upload - Comprehensive image analysis
- **EXIF Viewer**: https://exifdata.com/ - Extract image metadata
- **Steganography Online**: https://stylesuxx.github.io/steganography/ - LSB analysis

**Analysis Process:**
```
1. Visual inspection - Open image, look for unusual patterns
2. Metadata extraction - Go to https://exifdata.com/, upload image
3. LSB analysis - Go to https://georgeom.net/StegOnline/upload
4. File signature check - Use hex editor: https://hexed.it/
```

**What to look for:**
- Unusual pixel patterns in color planes
- Statistical anomalies in image data
- Metadata containing suspicious information
- File format inconsistencies

---

## Audio Steganography

### Common Hiding Techniques

**Metadata Hiding:**
Audio files contain ID3 tags where secrets can be stored.

**Example:**
```
Title: Summer Vacation
Artist: John Smith
Comment: ZmxhZ3thdWRpb19tZXRhZGF0YX0= (Base64 encoded flag)
```

**Spectral Hiding:**
Hidden images or text in the audio spectrogram - convert audio to visual representation!

**LSB Audio Hiding:**
Similar to image LSB, but with audio samples. Replace least significant bits of audio data with secret bits.

### Analysis Tools

**Online Tools:**
- **Online Spectrogram Generator**: https://convert.ing-now.com/audio-spectrogram-creator/
- **Metadata2go**: https://www.metadata2go.com/ - Extract audio metadata

**Analysis Process:**
1. Upload audio file to online spectrogram tool
2. Look for visual patterns in different frequency ranges
3. Check metadata using online tools
4. Listen carefully for any embedded sounds

---

## File Header Analysis

### Magic Bytes (File Signatures)

Every file type has a unique "signature" at the beginning:

```
PNG:  89 50 4E 47 (‰PNG)
JPEG: FF D8 FF
PDF:  25 50 44 46 (%PDF)
ZIP:  50 4B 03 04 (PK..)
GIF:  47 49 46 38 (GIF8)
```

**Example Detection:**
```
File: image.jpg
Claimed type: JPEG image
Actual header: 50 4B 03 04 (ZIP signature!)
Conclusion: This is a ZIP file disguised as JPEG!
```

**Online Tools:**
- **File Signatures Database**: https://www.filesignatures.net/
- **Online Hex Editor**: https://hexed.it/

### Polyglot Files

Files that are valid in multiple formats simultaneously!

**Example:** A file that's both a valid JPEG image AND a valid ZIP archive:
- View as JPEG: Shows a normal photo
- Open as ZIP: Contains hidden files with secret data

**Detection:**
- Check file with multiple programs
- Look for unusual file sizes
- Examine both beginning and end of file

---

## Metadata Extraction

### What Metadata Reveals

**Image Metadata:**
```
Camera Make: Apple iPhone 13
Date Taken: 2023-05-20 15:30:45
GPS Location: 40.7589, -73.9851 (New York City)
Custom Field: flag{gps_coordinates_leak}
```

**Audio Metadata:**
```
Artist: Unknown
Title: Recording_001
Comment: bmV3X2ZsYWdfe2F1ZGlvX21ldGF9 (Base64)
```

### Extraction Tools

**Online Metadata Extractors:**
- **Metadata2go**: https://www.metadata2go.com/ - Multi-format support
- **Jeffrey's EXIF Viewer**: http://exif.regex.info/exif.cgi - Detailed image analysis
- **ExifTool Online**: https://exif.tools/ - Comprehensive metadata extraction

---

## Tools Reference

### Essential Online Tools
- **StegOnline**: https://georgeom.net/StegOnline/upload - Image steganography analysis
- **Metadata2go**: https://www.metadata2go.com/ - Universal metadata extractor
- **Audio Spectrogram**: https://convert.ing-now.com/audio-spectrogram-creator/ - Audio visualization
- **EXIF Viewer**: https://exifdata.com/ - Image metadata extraction
- **Hex Editor**: https://hexed.it/ - Binary file analysis

### Key Techniques Summary
1. **Always check metadata first** - Often the easiest win
2. **Visual inspection** - Look for obvious anomalies
3. **File signature verification** - Ensure files are what they claim
4. **Spectral analysis** - For audio files, check the frequency domain
5. **Statistical analysis** - Look for data patterns that shouldn't exist