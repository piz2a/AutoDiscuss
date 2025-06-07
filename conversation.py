# Running a conversation between two LLMs
from ipykernel.eventloops import loop_qt

from llmapi_integration import LLMAPI


# The word 'discuss' will be used differently from 'converse' in this context.
# 'Discuss' will be used to refer to the process of getting the right answer through conversation.
def run_llm_conversation(model_a: LLMAPI, model_b: LLMAPI, initial_prompt: str, num_turns: int):
    # Conversation history as a list of strings
    dialogue = [initial_prompt]

    current_speaker = model_a

    # Keep track of all responses and timing
    conversation_log = []

    for turn in range(num_turns):
        # Call the current model with the current dialogue
        result = current_speaker.call(dialogue, first_is_system=True)

        # Extract response text and metadata
        response_text = result["response_text"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        duration = result["duration"]

        print(f"[Turn {turn + 1}/{num_turns}]", end=" ")

        # Log this turn
        conversation_log.append({
            "turn": turn + 1,
            "model": current_speaker.model,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": duration
        })

        # Append response to dialogue history
        remaining_turns = num_turns - turn - 1
        turn_annotation = f"\n\n남은 기회: {remaining_turns}번"
        if remaining_turns == 1:
            turn_annotation += "\n마지막 기회입니다. 결론을 내 주시기 바랍니다."
        dialogue.append(response_text + turn_annotation)

        # Alternate speaker
        current_speaker = model_b if current_speaker == model_a else model_a

    print()
    # After N turns, return the full conversation log
    return conversation_log, dialogue
