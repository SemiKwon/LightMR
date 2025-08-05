from utils.utils import *

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", use_fast=True)

class CustomDataset(Dataset):
    def __init__(self, video_dir, txt, processor, num_frames=16):
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames

        self.video_data = []
        with open(txt, 'r') as file:
            for line in file:
                part = line.strip().split('##')
                if len(part) != 2:
                    continue
                video_info, query = part
                video_info_parts = video_info.split()
                if len(video_info_parts) != 3:
                    continue
                video_name, start_time, end_time = (
                    video_info_parts[0],
                    float(video_info_parts[1]),
                    float(video_info_parts[2])
                )
                self.video_data.append((video_name, start_time, end_time, query))

        self.video_durations = {}
        self.prompts_with_conditions = [
            (
                "happens near the start of the video.",
                lambda qs, qe, Q1, Q2, Q3, Q4: qe <= Q1
            ),
            (
                "happens near the end of the video.",
                lambda qs, qe, Q1, Q2, Q3, Q4: qs >= Q3
            ),
            (
                "happens around the middle of the video.",
                lambda qs, qe, Q1, Q2, Q3, Q4: qs <= Q2 and qe >= Q2
            ),
            (
                "happens between the middle and the end.",
                lambda qs, qe, Q1, Q2, Q3, Q4: Q2 <= qs <= Q3
            ),
            (
                "happens between the start and the middle.",
                lambda qs, qe, Q1, Q2, Q3, Q4: Q1 <= qe <= Q2
            ),
        ]

    def extract_frames(self, video_path, num_frames):
        frames = []
        frame_times = []

        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                total_frames = stream.frames
                if total_frames == 0: 
                    total_frames = int(container.duration * stream.average_rate)

                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frame_indices_set = set(indices) 

                for i, frame in enumerate(container.decode(video=0)):
                    if i in frame_indices_set:
                        img = frame.to_image()
                        frames.append(img)
                        frame_times.append(frame.time)

                    if len(frames) >= num_frames:
                        break

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")

        return frames, frame_times

    def get_video_duration(self, video_name):
        if video_name in self.video_durations:
            return self.video_durations[video_name]

        video_path = os.path.join(self.video_dir, f'{video_name}.mp4')

        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base)
                self.video_durations[video_name] = duration
                return duration
        except Exception as e:
            print(f"Error retrieving duration for {video_name}: {e}")
            return 0.0

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_name, query_start, query_end, query = self.video_data[idx]
        video_path = os.path.join(self.video_dir, f'{video_name}.mp4')

        frames, _ = self.extract_frames(video_path, self.num_frames)
        if not frames:
            raise ValueError(f"No frames extracted from video {video_name}")

        video_duration = round(self.get_video_duration(video_name), 1)
        if video_duration == 0.0:
            raise ValueError(f"Video {video_name} has zero duration.")

        if query_end > video_duration:
            query_end = video_duration

        Q1, Q2, Q3, Q4 = (round(video_duration * 0.25, 1),
                           round(video_duration * 0.5, 1),
                           round(video_duration * 0.75, 1),
                           video_duration)

        applicable_prompts = []
        for i, (prompt_text, condition) in enumerate(self.prompts_with_conditions):
            if condition(query_start, query_end, Q1, Q2, Q3, Q4):
                applicable_prompts.append((i, prompt_text))

        if not applicable_prompts:
            raise ValueError(f"No prompt condition matched for video {video_name} "
                             f"(query_start={query_start}, query_end={query_end}).")
        
        selected_prompt = random.choice(applicable_prompts)[1]

        prompt = f"In this {video_duration:.0f}-second video, '{query}' {selected_prompt} Identify the start and end times."

        inputs = self.processor(
            images=frames,
            text=prompt,
            truncation=True,
            max_length=50,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True
        )

        label = f"It starts at {query_start:.0f} seconds and ends at {query_end:.0f} seconds."
        target_ids = self.processor.tokenizer(label, truncation=True, max_length=20, return_tensors="pt").input_ids

        return {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'pixel_values': inputs.pixel_values.squeeze(0),
            'labels': target_ids.squeeze(0)
        }