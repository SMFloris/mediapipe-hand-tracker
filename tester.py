import asyncio
import websockets

async def test_wss():
    uri = "ws://localhost:8765"  # Change port if needed
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WSS server!")
            
            # Receive response
            while True:
                response = await websocket.recv()
                print(f"📥 Received: {response}")

    except Exception as e:
        print(f"❌ WSS connection failed: {e}")

asyncio.run(test_wss())
