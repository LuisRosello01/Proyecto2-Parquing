import cv2
import numpy as np

def segmentar_matricula(image):
    # Verificar que la imagen no sea None
    if image is None:
        raise ValueError("La imagen proporcionada es inválida o está vacía.")
    
    # Resize the image while maintaining the aspect ratio
    height, width = image.shape[:2]
    new_width = 800  # Desired width
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    #if matricula_blava(resized_image):
    #    grayscale_image = cv2.bitwise_not(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY))
    #    resized_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    thresh = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    warped = corregir_perspectiva(thresh, resized_image)

    bottomhat_thresh = bottomhat(warped)
            
    # Save the bottomhat_thresh image for debugging purposes
    cv2.imwrite("bottomhat_thresh.jpg", bottomhat_thresh)
    cv2.imshow("Bottom-Hat Threshold", bottomhat_thresh)

    # Character segmentation
    contours_cleaned, _ = cv2.findContours(bottomhat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Visualize the image with contours
    visualization = cv2.cvtColor(bottomhat_thresh, cv2.COLOR_GRAY2BGR)
    for contour in contours_cleaned:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save or display the visualization
    #cv2.imshow("Contours Visualization", visualization)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    sorted_contours = sorted(contours_cleaned, key=lambda c: cv2.boundingRect(c)[0])
    character_images = []
    margin = 4
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(warped.shape[1] - x, w + 2 * margin)
        h = min(warped.shape[0] - y, h + 2 * margin)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 0.3 < aspect_ratio < 1.5 and area > 250 and h > 10:
            char_roi = warped[y:y+h, x:x+w]
            character_images.append(char_roi)

    return character_images

def matricula_blava(image):
    # Convert the license plate image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for blue color in HSV
    lower_blue = np.array([100, 50, 50])  # Adjust these values as needed
    upper_blue = np.array([140, 255, 255])

    # Create a mask for blue regions
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Calculate the percentage of blue pixels in the image
    blue_percentage = (np.sum(blue_mask > 0) / blue_mask.size) * 100

    # Set a threshold to determine if the license plate is blue
    is_blue_plate = blue_percentage > 10  # Adjust the threshold as needed

    # Display the result
    print(f"Percentage of blue pixels: {blue_percentage:.2f}%")
    return is_blue_plate

def corregir_perspectiva(thresh, resized_image):
    # Perspective correction
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    def order_points(pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        return np.array([tl, tr, br, bl], dtype="float32")

    ordered_box = order_points(box)
    width = int(np.max([
        np.linalg.norm(ordered_box[0] - ordered_box[1]),
        np.linalg.norm(ordered_box[2] - ordered_box[3])
    ]))
    height = int(np.max([
        np.linalg.norm(ordered_box[0] - ordered_box[3]),
        np.linalg.norm(ordered_box[1] - ordered_box[2])
    ]))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_box, dst)
    return cv2.warpPerspective(resized_image, M, (width, height))

def bottomhat(warped):
    # Bottom-Hat transformation
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bottomhat_kernel = np.ones((32, 32), np.uint8)
    bottomhat = cv2.morphologyEx(warped_gray, cv2.MORPH_BLACKHAT, bottomhat_kernel)
    _, bottomhat_thresh = cv2.threshold(bottomhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply erosion to the warped threshold image
    erosion_kernel = np.ones((3, 3), np.uint8)
    bottomhat_thresh = cv2.erode(bottomhat_thresh, erosion_kernel, iterations=1)
    
    # Use opening morphology to remove small noise
    #opening_kernel = np.ones((2, 1), np.uint8)
    #bottomhat_thresh = cv2.morphologyEx(bottomhat_thresh, cv2.MORPH_OPEN, opening_kernel)

    # Remove small noise points
    contours_noise, _ = cv2.findContours(bottomhat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtrar contornos pequeños que representan ruido
    #for contour in contours_noise:
        #area = cv2.contourArea(contour)
        #if 150 <= area <= 200:  # Ajustar el umbral para puntos medianos
            # Noise removal
            #bottomhat_thresh = eliminar_ruido(bottomhat_thresh)
    
    return bottomhat_thresh

def eliminar_ruido(bottomhat_thresh):
    erosion_kernel = np.ones((4, 3), np.uint8)
    eroded_image = cv2.erode(bottomhat_thresh, erosion_kernel, iterations=1)
    opening_kernel = np.ones((2, 1), np.uint8)
    cleaned_image = cv2.morphologyEx(eroded_image, cv2.MORPH_OPEN, opening_kernel)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    cleaned_image = cv2.bitwise_and(cleaned_image, mask)
    dilation_kernel = np.ones((3, 4), np.uint8)
    dilated = cv2.dilate(cleaned_image, dilation_kernel, iterations=1)
    return dilated

if __name__ == "__main__":
    # Example usage
    image_path = "flask/temp_matricula.jpg"
    image_path = r"deteccio_matricula\recortes\recorte_1.jpg"
    image = cv2.imread(image_path)
    characters = segmentar_matricula(image)
    #for i, char in enumerate(characters):
        #cv2.imshow(f'Character {i+1}', char)
    cv2.waitKey(0)
    cv2.destroyAllWindows()