# Extractable components

## AppHeader

- Source: `frontend/src/App.tsx`
- Category: layout
- Description: product identity, model configuration entry, account identity and logout.
- Extractable props: `providerName`, `model`, `username`.
- Hardcoded: product title, research positioning, controls and research-desk styling.

## ConversationList

- Source: `frontend/src/components/ConversationList.tsx`
- Category: layout
- Description: historical research conversations in the left rail.
- Extractable props: `activeId`, `conversations`.
- Hardcoded: action labels and list treatment.

## SourceSidebar

- Source: `frontend/src/components/SourceSidebar.tsx`
- Category: layout
- Description: live execution activity and traceable data evidence.
- Extractable props: `activities`, `citations`.
- Hardcoded: source card anatomy and status treatment.
