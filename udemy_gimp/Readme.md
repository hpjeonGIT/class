## Instructor: Chris Parker
## Class: GIMP Crash Course for Beginners!
- Gimp 2.10 or higher is recommended
- Basic image processing
- Adjusting contrast
    - Color -> Levels 
    - Adjust min/max with the range of histogram
- Healing
    - Tools -> Paint Tools -> Heal or from Tools Icon
    - ctrl+click to get nearby info
    - click or drag to the ROI to apply
- Scaling
    - 300DPI is too much for online-usage. 72DPI is recommended
- Zoom icon
    - ctrl for scroll-up/down
    - space for move
    
#############################################################

## Summary
- Title: 2D Game Artist: Learn GIMP For Game Art
- Instructors: Ben Tristem, GameDev.tv Team

## Section 1: Emotion icons - get creative at 8x8 pixels

4. Black and white outlines
- shift + ctrl + j: zoom full

8. A splash of color
- Flat shading: one color tone. 
- Dithering type effect: adding noise
- In HSV coloring, Value will change the amount of dark/brightness of the color

10. Using GIMP at high resolutions
- https://superuser.com/questions/1513000/how-to-enlarge-gimp-ui-font-and-icon-size-for-hdpi-screen

## Section 2: Game Icons - Creating 32x32 Pixel art

13. Working to a specification
- Binary choices
    - Looks real or iconised
    - 3D or 2D
    - Transparency required or not
    - Shading or not
    - Square edged or rounded or circular
    - Axi-symmetric/mirroring or not

14. Drawing Straight lines
- Click a cell. Then click shift key then move the cursor.
- Then click a target cell. It will draw a straight line

16. Using the erase
- Removing the background color to make it transparent
- Regular layers need to add alpha channel
- Then eraser will clean the background color, making the layer transparent

18. Using the paintbrush and airbrush
- 9 brushes

20. Filling an area with one color
- Using bucket
    - To bucket on a trasparent layer, the below layer must have colors/shapes
    - Will not fill on a transparent layer - or need the option of Fill transparent areas
    - If the below layer is transparent but has an area, then the mapping area can be filled using the option of Sample merged

21. Using the selection tools
- Fuzzy select: select contiguous pixels 
    - To select pixels of different layers, enable `Sample merged`

22. Shading with Gradients
- Blend in 2.8 (?) Gradient in 2.10


31. Floating Selections
- GIMP seems not responding but it could be in the mode of floating selection
- When select is applied with float, a new layer is produced
- Layer->Anchor Layer or Anchor icon will merge the floating selection into the base layer
- Or make a new layer then the float layer will be migrated into it
- Any copy/paste will produce a floating selection

## Section 3: Animated Sprites

## Section 4: Character Sketch

66. Drawing options
- wheel + ctrl to zoom in/out

68. The paths tool
- To delete path, delete key doesn't work. In the paths tab, find the path and delete it
