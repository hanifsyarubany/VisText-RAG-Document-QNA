import chainlit as cl
import requests
import base64

BACKEND_URL = "http://localhost:8000/chat-inference"

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "👋 Welcome to the Multimodal RAG Chatbot Demo!\n\n"
            "You can ask natural language questions, and I'll retrieve both **text and image** information to answer.\n\n"
            "If relevant, I’ll also show a related image to help you better understand the answer!"
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content

    try:
        response = requests.post(BACKEND_URL, json={"user_query": user_query})
    except Exception:
        await cl.Message(content="❌ Gagal menghubungi server.").send()
        return

    if response.status_code != 200:
        await cl.Message(content="❌ Terjadi kesalahan di server.").send()
        return

    result = response.json()
    content = result.get("response", "")
    need_image = result.get("need_image", False)
    img_base64 = result.get("img_base64", "")


    # Image response (attached as element)
    if need_image and img_base64:
        try:
            if img_base64.startswith("data:image"):
                img_base64 = img_base64.split(",")[1]

            image_bytes = base64.b64decode(img_base64)

            await cl.Message(
                content= content,
                elements=[
                    cl.Image(
                        name="result.jpg",
                        display="inline",
                        content=image_bytes,
                        size="large"
                    )
                ]
            ).send()
        except Exception as e:
            print("Image decoding error:", e)
            await cl.Message(content="⚠️ Failed to render image.").send()
    else:
        # Text response
        await cl.Message(content=content).send()
