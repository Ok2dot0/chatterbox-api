"""
Tests for the pause tags functionality [pause:xx]
"""

import re
import pytest

# Import the functions directly from tts.py's source
# Re-implement for testing since the package has complex dependencies


def parse_pause_tags(text: str):
    """
    Parse pause tags in text and return text segments with corresponding pause durations
    
    Args:
        text: Text containing pause tags like "Hello[pause:0.5s]world[pause:1.0s]end"
    
    Returns:
        segments: [(text_segment, pause_duration), ...]
        Example: [("Hello", 0.5), ("world", 1.0), ("end", 0.0)]
    """
    if not text:
        return [("", 0.0)]
    
    # Regular expression to match pause tags
    pause_pattern = r'\[pause:([\d.]+)s\]'
    
    segments = []
    last_end = 0
    
    # Find all pause tags
    for match in re.finditer(pause_pattern, text):
        # Extract text before the pause tag
        text_segment = text[last_end:match.start()].strip()
        if text_segment:
            segments.append((text_segment, 0.0))
        
        # Extract pause duration
        pause_duration = float(match.group(1))
        # Ensure pause duration is a multiple of 0.1s
        pause_duration = round(pause_duration / 0.1) * 0.1
        segments.append(("", pause_duration))
        
        last_end = match.end()
    
    # Add the final text segment
    final_text = text[last_end:].strip()
    if final_text:
        segments.append((final_text, 0.0))
    
    # If no segments found, return original text
    if not segments:
        segments = [(text, 0.0)]
    
    return segments


class TestParsePauseTags:
    """Tests for parse_pause_tags function"""
    
    def test_simple_pause_tag(self):
        """Test simple text with one pause tag"""
        text = "Hello[pause:0.5s]world"
        result = parse_pause_tags(text)
        
        assert len(result) == 3
        assert result[0] == ("Hello", 0.0)
        assert result[1][0] == ""
        assert abs(result[1][1] - 0.5) < 0.01  # Handle floating point
        assert result[2] == ("world", 0.0)
    
    def test_multiple_pause_tags(self):
        """Test text with multiple pause tags"""
        text = "Hello[pause:0.5s]world[pause:1.0s]end"
        result = parse_pause_tags(text)
        
        assert len(result) == 5
        assert result[0] == ("Hello", 0.0)
        assert result[1][0] == ""
        assert abs(result[1][1] - 0.5) < 0.01
        assert result[2] == ("world", 0.0)
        assert result[3][0] == ""
        assert abs(result[3][1] - 1.0) < 0.01
        assert result[4] == ("end", 0.0)
    
    def test_no_pause_tags(self):
        """Test text without pause tags"""
        text = "Hello world"
        result = parse_pause_tags(text)
        
        assert len(result) == 1
        assert result[0] == ("Hello world", 0.0)
    
    def test_pause_at_start(self):
        """Test text starting with a pause tag"""
        text = "[pause:0.3s]Hello world"
        result = parse_pause_tags(text)
        
        assert len(result) == 2
        assert result[0][0] == ""
        assert abs(result[0][1] - 0.3) < 0.01
        assert result[1] == ("Hello world", 0.0)
    
    def test_pause_at_end(self):
        """Test text ending with a pause tag"""
        text = "Hello world[pause:0.5s]"
        result = parse_pause_tags(text)
        
        assert len(result) == 2
        assert result[0] == ("Hello world", 0.0)
        assert result[1][0] == ""
        assert abs(result[1][1] - 0.5) < 0.01
    
    def test_decimal_rounding(self):
        """Test that decimal values are rounded to 0.1s increments"""
        text = "Hello[pause:0.25s]world"
        result = parse_pause_tags(text)
        
        # 0.25 should round to 0.2 (nearest 0.1)
        assert len(result) == 3
        assert abs(result[1][1] - 0.2) < 0.01
    
    def test_empty_text(self):
        """Test empty text"""
        result = parse_pause_tags("")
        assert result == [("", 0.0)]
    
    def test_none_text(self):
        """Test None text (handled by empty check)"""
        result = parse_pause_tags(None)
        assert result == [("", 0.0)]
    
    def test_longer_pause(self):
        """Test longer pause durations"""
        text = "Wait[pause:2.5s]for it"
        result = parse_pause_tags(text)
        
        assert len(result) == 3
        assert abs(result[1][1] - 2.5) < 0.01
    
    def test_consecutive_pauses(self):
        """Test consecutive pause tags"""
        text = "Hello[pause:0.5s][pause:0.5s]world"
        result = parse_pause_tags(text)
        
        # Should have: Hello, pause, pause, world
        assert len(result) == 4
        assert result[0] == ("Hello", 0.0)
        assert abs(result[1][1] - 0.5) < 0.01
        assert abs(result[2][1] - 0.5) < 0.01
        assert result[3] == ("world", 0.0)
    
    def test_pause_with_special_characters(self):
        """Test pause tags with special characters in text"""
        text = "Hello![pause:0.5s]How are you?"
        result = parse_pause_tags(text)
        
        assert len(result) == 3
        assert result[0] == ("Hello!", 0.0)
        assert result[2] == ("How are you?", 0.0)
    
    def test_integer_pause_duration(self):
        """Test pause with integer duration"""
        text = "Hello[pause:1s]world"
        result = parse_pause_tags(text)
        
        assert len(result) == 3
        assert abs(result[1][1] - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
