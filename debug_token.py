import os
from dotenv import load_dotenv

# Load .env explicitly
load_dotenv(override=True)

token = os.environ.get("TELEGRAM_BOT_TOKEN")

print("----- TOKEN DEBUGGER -----")
if token is None:
    print("❌ Error: Token is NONE. (.env file error)")
else:
    # Check for quotes
    if token.startswith('"') or token.startswith("'"):
        print(f"⚠️ Warning: Token starts with QUOTE: {token[0]}")
    if token.endswith('"') or token.endswith("'"):
        print(f"⚠️ Warning: Token ends with QUOTE: {token[-1]}")
        
    # Check for whitespace
    if token.strip() != token:
        print("⚠️ Warning: Token has leading/trailing WHITESPACE!")
        print(f"   Original: '{token}'")
        print(f"   Stripped: '{token.strip()}'")
    
    # Check length
    print(f"✅ Token Length: {len(token)}")
    print(f"✅ First 5 chars: {token[:5]}")
    print(f"✅ Last 5 chars: {token[-5:]}")

    # Validate format
    parts = token.strip().split(':')
    if len(parts) != 2:
        print("❌ Error: Invalid Format (Missing ':' or too many parts)")
    elif not parts[0].isdigit():
        print("❌ Error: First part is not numbers")
    else:
        print("✅ Format looks good (Number:String)")

print("--------------------------")