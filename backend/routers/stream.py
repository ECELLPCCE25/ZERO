import cv2
import requests
import numpy as np
import time
import io
import base64  # Import base64 for decoding

# API endpoint details for the analysis service
API_URL = "http://43.228.115.146:8000/gpu/analyze_frame_stateless/"
API_HEADERS = {"accept": "application/json"}

# Initialize video capture from a file
# Ensure the path '../dataset/1.mp4' is correct relative to where you run the script
video_path = "../dataset/1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source at {video_path}")
    print("Please check the file path or if the file exists.")
    exit()

print(f"Accessing video file: {video_path}. Sending frames to {API_URL}")

# Variable to store the previous frame for the API call
prev_frame = None
frame_count = 0

# Start the main loop to read frames from the video
while True:
    # Read a frame from the video source
    # ret is a boolean indicating if the frame was read successfully
    # frame is the image data (NumPy array)
    ret, frame = cap.read()

    # If the frame was not read successfully (e.g., end of video), break the loop
    if not ret:
        print("End of video stream or error reading frame. Exiting ...")
        break

    # --- API Call Preparation ---
    # We need a previous frame for the API call.
    # For the very first frame, we use the current frame as the previous one.
    if prev_frame is None:
        # Make a copy to ensure 'prev_frame' is a distinct image from 'frame'
        prev_frame = frame.copy()

    # Convert the previous and current frames to JPEG format in memory
    # cv2.imencode compresses the image and returns a tuple: (success, buffer)
    # We specify the format ('.jpg') and optionally the quality (0-100)
    _, prev_frame_encoded = cv2.imencode(
        ".jpg", prev_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    _, current_frame_encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    # Convert the encoded frames (numpy arrays) to raw bytes
    prev_frame_bytes = prev_frame_encoded.tobytes()
    current_frame_bytes = current_frame_encoded.tobytes()

    # Prepare the files dictionary for the multipart/form-data request to the API
    # The keys ('prev_frame', 'current_frame') should match the API's expected form field names
    # The value is a tuple: (filename, file_like_object, content_type)
    # io.BytesIO treats the bytes in memory as a file
    files = {
        "prev_frame": ("prev_frame.jpg", io.BytesIO(prev_frame_bytes), "image/jpeg"),
        "current_frame": (
            "current_frame.jpg",
            io.BytesIO(current_frame_bytes),
            "image/jpeg",
        ),
    }

    # --- API Call Execution ---
    api_response_data = None
    try:
        # Send the POST request to the analysis API
        response = requests.post(API_URL, headers=API_HEADERS, files=files)

        # Process the API response
        if response.status_code == 200:
            # If the request was successful, parse the JSON response
            api_response_data = response.json()
            # Uncomment the line below to print the raw API response data for each frame
            # print(f"API Response for frame {frame_count}: {api_response_data}")

            # --- Visualization Logic ---
            # First, try to use the annotated frame if provided by the API
            if api_response_data and "annotated_frame_base64" in api_response_data:
                try:
                    img_bytes = io.BytesIO(
                        base64.b64decode(api_response_data["annotated_frame_base64"])
                    )
                    # Convert bytes to numpy array
                    img_array = np.frombuffer(img_bytes.read(), np.uint8)
                    # Decode numpy array into image
                    decoded_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if decoded_frame is not None:
                        frame = (
                            decoded_frame  # Use the decoded frame for further drawing
                        )
                    else:
                        print(
                            f"Warning: Could not decode annotated frame for frame {frame_count}. Drawing on original frame."
                        )
                except Exception as e:
                    print(
                        f"Error decoding annotated frame base64 for frame {frame_count}: {e}"
                    )
                    # Continue drawing on the original 'frame' if decoding fails

            # Now, draw additional information from the API response onto the 'frame'

            # Draw Danger Regions (Rectangles)
            if api_response_data and "danger_regions" in api_response_data:
                for region in api_response_data["danger_regions"]:
                    try:
                        # Assuming region is a dict with 'x', 'y', 'w', 'h'
                        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                        # Ensure coordinates are within frame bounds if necessary
                        # cv2.rectangle(image, start_point, end_point, color, thickness)
                        # Color is BGR (Blue, Green, Red)
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (0, 0, 255), 2
                        )  # Red rectangle, thickness 2
                    except KeyError as e:
                        print(
                            f"Warning: Missing key in danger_regions data for frame {frame_count}: {e}"
                        )
                    except Exception as e:
                        print(
                            f"Error drawing danger region for frame {frame_count}: {e}"
                        )

            # Draw Dangerous Cells (Requires Grid Information - Skipping for now)
            # If you know the grid cell size and origin, you can uncomment and modify this
            # if api_response_data and 'dangerous_cells' in api_response_data:
            #     cell_size_w = 50 # Example: replace with actual cell width
            #     cell_size_h = 50 # Example: replace with actual cell height
            #     grid_origin_x = 0 # Example: replace with actual grid origin x
            #     grid_origin_y = 0 # Example: replace with actual grid origin y
            #     for cell in api_response_data['dangerous_cells']:
            #         try:
            #             row, col = cell['row'], cell['col']
            #             x = grid_origin_x + col * cell_size_w
            #             y = grid_origin_y + row * cell_size_h
            #             cv2.rectangle(frame, (x, y), (x + cell_size_w, y + cell_size_h), (0, 165, 255), -1) # Orange filled rectangle
            #         except KeyError as e:
            #              print(f"Warning: Missing key in dangerous_cells data for frame {frame_count}: {e}")
            #         except Exception as e:
            #              print(f"Error drawing dangerous cell for frame {frame_count}: {e}")

            # Draw Convergence Cells (Requires Grid Information - Skipping for now)
            # If you know the grid cell size and origin, you can uncomment and modify this
            # if api_response_data and 'convergence_cells' in api_response_data:
            #     cell_size_w = 50 # Example: replace with actual cell width
            #     cell_size_h = 50 # Example: replace with actual cell height
            #     grid_origin_x = 0 # Example: replace with actual grid origin x
            #     grid_origin_y = 0 # Example: replace with actual grid origin y
            #     for cell in api_response_data['convergence_cells']:
            #         try:
            #             row, col = cell['row'], cell['col']
            #             x = grid_origin_x + col * cell_size_w
            #             y = grid_origin_y + row * cell_size_h
            #             cv2.rectangle(frame, (x, y), (x + cell_size_w, y + cell_size_h), (255, 255, 0), -1) # Cyan filled rectangle
            #         except KeyError as e:
            #              print(f"Warning: Missing key in convergence_cells data for frame {frame_count}: {e}")
            #         except Exception as e:
            #              print(f"Error drawing convergence cell for frame {frame_count}: {e}")

            # Draw Predicted Convergence Points (Circles)
            if (
                api_response_data
                and "predicted_convergence_points" in api_response_data
            ):
                for point in api_response_data["predicted_convergence_points"]:
                    try:
                        # Assuming point is a dict with 'x', 'y'
                        x, y = point["x"], point["y"]
                        # Ensure coordinates are within frame bounds if necessary
                        # cv2.circle(image, center_coordinates, radius, color, thickness)
                        cv2.circle(
                            frame, (x, y), 5, (0, 255, 0), -1
                        )  # Green filled circle, radius 5
                    except KeyError as e:
                        print(
                            f"Warning: Missing key in predicted_convergence_points data for frame {frame_count}: {e}"
                        )
                    except Exception as e:
                        print(
                            f"Error drawing predicted convergence point for frame {frame_count}: {e}"
                        )

            # Display Person Count (Text)
            if api_response_data and "person_count" in api_response_data:
                try:
                    count = api_response_data["person_count"]
                    # cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)
                    cv2.putText(
                        frame,
                        f"Persons: {count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )  # Green text
                except KeyError as e:
                    print(
                        f"Warning: Missing key 'person_count' in API response for frame {frame_count}: {e}"
                    )
                except Exception as e:
                    print(f"Error displaying person count for frame {frame_count}: {e}")

            # Quiver plot visualization would typically require decoding the base64
            # and potentially overlaying it or displaying in a separate window.
            # Decoding example (you would need to decide how to display/overlay):
            # if api_response_data and 'quiver_plot_base64' in api_response_data:
            #     try:
            #         quiver_img_bytes = io.BytesIO(base64.b64decode(api_response_data['quiver_plot_base64']))
            #         quiver_img_array = np.frombuffer(quiver_img_bytes.read(), np.uint8)
            #         quiver_image = cv2.imdecode(quiver_img_array, cv2.IMREAD_COLOR)
            #         if quiver_image is not None:
            #             # Example: Display in a separate window
            #             cv2.imshow("Quiver Plot", quiver_image)
            #             # Example: Overlay (requires resizing/positioning)
            #             # frame[y:y+h, x:x+w] = quiver_image # Simple overlay, assumes size match
            #     except Exception as e:
            #         print(f"Error decoding or displaying quiver plot for frame {frame_count}: {e}")

        else:
            # Print an error if the API request failed
            print(
                f"API request failed for frame {frame_count} with status code {response.status_code}"
            )
            print(f"Response body: {response.text}")

    except requests.exceptions.RequestException as e:
        # Catch and print exceptions that occur during the requests call (e.g., connection errors, timeouts)
        print(f"Error sending request to API for frame {frame_count}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during API call or processing
        print(
            f"An unexpected error occurred during API call or processing for frame {frame_count}: {e}"
        )

    # --- Display the Frame ---
    # Display the current frame in a window named 'Video Feed'
    # This frame will include any visualizations added based on the API response
    cv2.imshow("Video Feed", frame)

    # --- Loop Control ---
    # Update prev_frame for the next iteration of the loop
    # The current frame becomes the 'previous' frame for the next API call
    prev_frame = frame.copy()

    frame_count += 1
    # Optional: Print frame count to track progress
    # print(f"Processed frame {frame_count}")

    # Wait for 1 millisecond (or longer) and check for key press
    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
print("Video capture released and windows closed.")
