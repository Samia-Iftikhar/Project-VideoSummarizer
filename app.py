from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
import torch
import cv2
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip
import gradio as gr

model_id = "microsoft/Phi-3.5-vision-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4
)

def extract_audio(video_path):
    """
    Extracts audio from a video file and saves it as an MP3 file.

    Parameters:
    video_path (str): The path to the video file.
    audio_output_path (str): The path where the extracted audio will be saved.
    """
    try:
        audio_output_path = "temp_audio.mp3"
        # Load the video file
        video = VideoFileClip(video_path)

        # Extract audio from the video
        audio = video.audio

        # Save the audio to the specified path
        audio.write_audiofile(audio_output_path)

        return audio_output_path

    except Exception as e:
        print(f"An error occurred: {e}")

def transcribe_audio(audio_path, model_size="medium", language="en"):
    """
    Transcribes an audio file using the Whisper model.

    Parameters:
    audio_path (str): The path to the audio file (e.g., MP3 or WAV).
    model_size (str): The size of the Whisper model to use. Options: "tiny", "base", "small", "medium", "large".
    language (str): The language of the audio. Default is "en" for English.

    Returns:
    str: The transcribed text from the audio.
    """
    # Load Whisper model
    model = whisper.load_model(model_size)

    # Transcribe the audio
    result = model.transcribe(audio_path, language=language)

    # Return the transcription
    return result['text']

def extract_key_frames(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    # Get the original frames per second (FPS) of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval to extract frames based on the target FPS
    interval = int(original_fps // target_fps)

    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break

        # Save one frame per interval
        if frame_count % interval == 0:
            # Convert frame from BGR (OpenCV) to RGB (PIL format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_count += 1

    cap.release()
    return frames

# Summarize the video
def summarize_video(video_path, target_fps):

    # Extract frames from the video
    frames = extract_key_frames(video_path, target_fps)
    images = []
    placeholder = ""

    for i, frame in enumerate(frames):
    # Add the frame to the images list
        images.append(frame)
    # Create the placeholder for each image
        placeholder += f"<|image_{i+1}|>\n"

    messages = [
        {"role": "user", "content": placeholder+"What is shown in these video frames? Be extremely detailed and specific."},
        ]

    prompt = processor.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda")

    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    torch.cuda.empty_cache()

# remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False)[0]

    return response

def summarize_text(text, model="Falconsai/text_summarization"):
    """
    Summarizes a given text using a Hugging Face summarization model.

    Parameters:
    text (str): The text to be summarized.
    model (str): The Hugging Face model to use for summarization. Default is "Falconsai/text_summarization".

    Returns:
    str: The summarized text.
    """
    from transformers import pipeline

    # Use the summarization pipeline with the specified model
    summarizer = pipeline("summarization", model=model)

    # Generate summary
    summary = summarizer(text)

    # Extract and return the summarized text
    return summary[0]['summary_text']


def combine_and_summarize(video_path, model_size="medium", language="en"):
    """
    Combines transcription from audio and extracted text from multimodal input, then summarizes the combined text.

    Parameters:
    multimodal_input (str): Path to the multimodal input (e.g., image).
    model_size (str): Size of the Whisper model to use for transcription.
    language (str): Language of the audio for transcription.

    Returns:
    str: The summarized text after combining transcription and multimodal input.
    """
    try:
      # Step 1: Extract audio using Moviepy
      audio_path = extract_audio(video_path)

      # Step 2: Transcribe the audio using Whisper
      try:
          transcription_text = transcribe_audio(audio_path, model_size, language)
      except Exception as e:
          transcription_text = ""  # Set to empty string if transcription fails

      # Step 3: Extract text from the multimodal model (e.g., image or video captioning)
      multimodal_text = summarize_video(video_path, target_fps=0.25)

      # Step 4: Combine the texts (only if transcription is available)
      if transcription_text:
          combined_text = transcription_text + "\n" + multimodal_text
      else:
          combined_text = multimodal_text

      # Step 5: Summarize the combined text
      summarized_text = summarize_text(combined_text)

      return summarized_text
    except RuntimeError as e:
      if "CUDA out of memory" in str(e):
        return "CUDA out of memory error occurred. Try a shorter video."
      else:
        return f"An error occurred: {e}"


demo = gr.Interface(
    fn=combine_and_summarize,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Textbox(label="Summarized Text")
)

demo.launch(debug=True)