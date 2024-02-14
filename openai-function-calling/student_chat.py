from openai import OpenAI
import json

client = OpenAI()


def apply_for_leave(number_of_days, reason, type_of_leave):
    """
    A function which calls the internal API to apply for leave
    :param number_of_days: Number of days required
    :param reason: Reason for leave
    :param type_of_leave: Sick or Holiday leave
    :return: Returns status for the leave
    """
    # Call the internal API for apply for the leave
    return json.dumps({"days": number_of_days, "reason": reason, "type": type_of_leave, "status": "approved"})


def get_my_marks(registration_number):
    """
    Returns the marks for the given registration number
    :param registration_number: Registration id of the student
    :return: Returns the marks
    """
    # Call the internal API to get the marks
    return json.dumps({"registration_number": registration_number, "marks": {"CS": 90, "English": 89}})


def run_conversation(message):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": message}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_for_leave",
                "description": "A function which calls the internal API to apply for leave ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number_of_days": {
                            "type": "integer",
                            "description": "Number of days required",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for leave",
                        },
                        "type_of_leave": {"type": "string", "enum": ["Sick Leave", "Holiday Leave"]},
                    },
                    "required": ["number_of_days", "reason", "type_of_leave"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_my_marks",
                "description": "Returns the marks for the given registration number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "registration_number": {
                            "type": "integer",
                            "description": "Registration id of the student",
                        }
                    },
                    "required": ["registration_number"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "apply_for_leave": apply_for_leave,
            "get_my_marks": get_my_marks,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "apply_for_leave":
                function_response = function_to_call(
                    number_of_days=function_args.get("number_of_days"),
                    reason=function_args.get("reason"),
                    type_of_leave=function_args.get("type_of_leave")
                )
            elif function_name == "get_my_marks":
                function_response = function_to_call(
                    registration_number=function_args.get("registration_number"),
                )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


if __name__ == '__main__':
    # print(run_conversation("I am not well today. I want to apply for leave today").choices[0].message.content)
    print(run_conversation("My registration number is 1000. What is my marks?").choices[0].message.content)
