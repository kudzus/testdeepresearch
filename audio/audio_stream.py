import threading
import time
import numpy as np
import sounddevice as sd

class AudioStream:
    def __init__(self, samplerate=16000, channels=1, dtype='int16', buffer_duration=5.0):
        """
        Continuous AudioStream using a ring buffer for multiple consumers.
        
        :param samplerate: Sample rate in Hz (e.g., 16000 for speech recognition)
        :param channels: Number of audio channels (1 for mono)
        :param dtype: Data type (e.g., 'int16' for 16-bit PCM)
        :param buffer_duration: Duration in seconds to keep in the ring buffer
        """
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.buffer_duration = buffer_duration
        self.capacity = int(samplerate * buffer_duration)
        self.ring_buffer = np.zeros((self.capacity, channels), dtype=self.dtype)
        
        self.write_index = 0
        self.total_written = 0
        self.stopped = False
        self.lock = threading.Lock()

        # Dictionary mapping consumer_id -> last_read pointer
        self.last_reads = {}

        # Start capturing audio using sounddevice
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal method called automatically by sounddevice for new audio samples."""
        if status:
            print("Audio Status:", status)
        with self.lock:
            num_frames = indata.shape[0]
            # If indata has more frames than the capacity, only keep the last portion
            if num_frames > self.capacity:
                indata = indata[-self.capacity:]
                num_frames = self.capacity

            end_index = self.write_index + num_frames
            if end_index <= self.capacity:
                self.ring_buffer[self.write_index:end_index] = indata
            else:
                first_part = self.capacity - self.write_index
                self.ring_buffer[self.write_index:] = indata[:first_part]
                self.ring_buffer[0:num_frames - first_part] = indata[first_part:]
            self.write_index = (self.write_index + num_frames) % self.capacity
            self.total_written += num_frames

    def register_consumer(self, consumer_id):
        """
        Register a new consumer with its own read pointer.
        
        :param consumer_id: A unique string or ID to track the consumer
        """
        with self.lock:
            self.last_reads[consumer_id] = 0

    def get_new_audio_chunks(self, consumer_id):
        """
        Return new audio samples for a specific consumer, updating the consumer’s read pointer.
        
        :param consumer_id: The unique ID of the consumer calling this method
        :return: np.array of shape (N, channels) containing new samples
        """
        with self.lock:
            if consumer_id not in self.last_reads:
                return np.empty((0, self.channels), dtype=self.dtype)

            new_count = self.total_written - self.last_reads[consumer_id]
            if new_count <= 0:
                return np.empty((0, self.channels), dtype=self.dtype)

            if new_count > self.capacity:
                self.last_reads[consumer_id] = self.total_written - self.capacity
                new_count = self.capacity

            start_index = self.last_reads[consumer_id] % self.capacity
            end_index = self.total_written % self.capacity

            if start_index < end_index:
                new_data = self.ring_buffer[start_index:end_index].copy()
            else:
                new_data = np.concatenate(
                    (self.ring_buffer[start_index:], self.ring_buffer[:end_index]),
                    axis=0
                )
            self.last_reads[consumer_id] = self.total_written
            return new_data

    def audio_generator(self, consumer_id, yield_interval=0.1):
        """
        Generator that yields raw PCM bytes for a specific consumer.
        """
        while not self.stopped:
            chunk = self.get_new_audio_chunks(consumer_id)
            if chunk.size > 0:
                yield chunk.tobytes()
            time.sleep(yield_interval)

    def get_audio_data(self):
        """
        Return the entire continuous audio data from the ring buffer.
        """
        with self.lock:
            if self.total_written < self.capacity:
                return self.ring_buffer[:self.write_index].copy()
            else:
                return np.concatenate(
                    (self.ring_buffer[self.write_index:], self.ring_buffer[:self.write_index]),
                    axis=0
                )

    def stop(self):
        """Stop the audio stream."""
        self.stream.stop()
        self.stream.close()
        self.stopped = True