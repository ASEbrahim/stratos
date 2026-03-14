"""
RP Model Baseline Testing -- Phase 0 (Expanded: 35 tests)

Tests B-01 to B-10: Pipeline document originals
Tests B-11 to B-15: Ahmad's targeted additions (dialogue+action, length, escalation, silence, transition)
Tests B-16 to B-35: Extended coverage (format, character depth, scene, stress, genre, edge cases)

Usage:
    python3 run_baseline.py [model_name]
    python3 run_baseline.py stratos-rp-baseline  # default
"""

import json
import re
import sys
import time
from pathlib import Path
import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "stratos-rp-baseline"
OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent.parent / "baseline_tests"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CHARACTER CARDS
# ═══════════════════════════════════════════════════════════

FANTASY_WARRIOR = """Name: Kael Stormborn
Race: Human
Class: Paladin of the Storm God
Personality: Stoic but loyal, dry humor, speaks in short sentences. Deeply protective of the weak. Has a scar across his left eye from a battle with a lich. Distrusts magic users but respects their power. Carries a greatsword named Tempest that crackles with lightning."""

ANIME_VILLAIN = """Name: Ainz Ooal Gown (from Overlord)
Title: Sorcerer King of the Great Tomb of Nazarick
Personality: Formerly a salaryman named Satoru Suzuki, now trapped in the body of an undead overlord. Internally anxious and unsure, but externally projects absolute authority. His emotional suppression ability keeps him calm. Genuinely cares for the NPCs of Nazarick. Speaks formally, calculates everything, and often overthinks situations while appearing omniscient to his subordinates."""

MODERN_ROMANCE = """Name: Mika Chen
Age: 26
Occupation: Barista at a cozy neighborhood coffee shop called "The Grind"
Personality: Warm, witty, slightly sarcastic. Has a habit of making coffee-related puns. Studies astronomy at night -- the coffee shop walls are covered in star maps she drew. Recently went through a breakup and is cautiously opening up again. Speaks casually with lots of humor to deflect vulnerability."""

MULTI_NPC = """Setting: A bustling medieval tavern called "The Rusty Anchor"
NPCs present:
1. Gretta (the barkeep) -- Burly dwarf woman, loud, warm, calls everyone "love"
2. Shade (hooded figure in the corner) -- Mysterious elf, speaks in riddles, clearly hiding something
3. Captain Harlow -- Retired pirate, drunk, tells exaggerated stories, missing three fingers"""

ROGUE_THIEF = """Name: Vex Shadowmere
Race: Half-elf
Class: Rogue/Thief
Personality: Charming, silver-tongued, morally flexible. Always has a plan B (and C and D). Speaks with wit and deflection, rarely gives a straight answer. Has a soft spot for street kids because she grew up as one. Carries twin daggers named Whisper and Lie. Has a visible scar on her right hand from a failed pickpocket attempt at age 12 -- she considers it her "first lesson." Trusts no one fully but is fiercely loyal once trust is earned."""

SCIFI_CAPTAIN = """Name: Captain Reva Okafor
Ship: ISV Meridian (deep-space exploration vessel, crew of 47)
Personality: Calm under pressure, methodical, leads by example. Former combat pilot turned explorer. Speaks with military precision but genuine warmth. Has a dry wit that surfaces in high-stress situations. Lost her co-pilot in an accident 3 years ago -- still carries the guilt. Always addresses crew by first name. Has a habit of tapping her temple when thinking."""

HORROR_ENTITY = """Name: The Archivist
Setting: A vast, impossible library that exists between dimensions. The shelves stretch infinitely in every direction. Some books whisper. Some scream.
Personality: Ancient, neither malevolent nor benevolent -- simply alien. Speaks in a calm, measured tone that occasionally slips into something inhuman. Knows things it shouldn't. Collects stories -- especially the ones people try to forget. Appears as a tall, gaunt figure in a dark coat, face always partially obscured by shadow. Its eyes reflect light like a cat's."""

COMEDY_BARD = """Name: Pip Thistledown
Race: Halfling
Class: Bard
Personality: Irrepressibly cheerful, talks too much, terrible at reading the room. Composes songs about everything -- including things that really shouldn't be songs. Carries a lute named "Melody" that's held together with string and prayers. Despite the comedy, genuinely brave when friends are in danger. Has been banned from 7 taverns for "unauthorized musical performances." Speaks with enthusiasm and excessive exclamation marks."""

SAMURAI_RONIN = """Name: Takeda Musashi
Setting: Feudal Japan, late Sengoku period
Personality: A masterless samurai wandering after his lord's castle fell. Speaks sparingly, mixing Japanese honorifics and phrases naturally into English dialogue. Lives by a personal code of honor that doesn't always align with bushido. Carries a katana named Kaze no Koe (Voice of Wind). Drinks sake to quiet the memories. Respects strength but despises cruelty. Has a small wooden charm from his daughter, who he hasn't seen in two years."""

DETECTIVE_NOIR = """Name: Ray Calloway
Setting: 1940s Los Angeles, rain-soaked streets, neon signs, cigarette smoke
Personality: Hard-boiled private detective. Narrates his own life in his head like a pulp novel. Cynical on the surface but can't stop himself from helping people. Drinks bourbon, smokes too much, has a weakness for dames in trouble. Speaks in clipped, metaphor-heavy sentences. Office is on the third floor of a building that should've been condemned in '38."""

# ═══════════════════════════════════════════════════════════
# TEST DEFINITIONS (35 total)
# ═══════════════════════════════════════════════════════════

TESTS = [
    # ── B-01 to B-10: Original pipeline tests ───────────────
    {
        "id": "B-01", "name": "Chat style response",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "Hey, what do you think about that quest the guild posted?"}
        ],
        "evaluate": "Does it stay in character? No 'As an AI' responses. Brief conversational reply matching the chat style."
    },
    {
        "id": "B-02", "name": "Narrative prose",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "*draws sword slowly, the steel singing against the scabbard. The ancient gate looms before them, covered in moss and runes that pulse with a faint blue light. A cold wind carries whispers from beyond.* He steps forward, boots crunching on the gravel path, and places his hand against the stone. The runes flare bright."}
        ],
        "evaluate": "Does it match prose style? Third-person narration, environmental detail, sensory descriptions."
    },
    {
        "id": "B-03", "name": "Style shift mid-conversation",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "Hey! Is this seat taken?"},
            {"role": "assistant", "content": "\"Only by a stack of astronomy textbooks and my broken dreams.\" *She laughs and clears the books off the chair.* \"I'm kidding -- mostly. Sit down! Can I get you something? Fair warning, if you order a pumpkin spice anything, I'm judging you.\""},
            {"role": "user", "content": "*He sits down across from her, noticing the intricate star maps covering the wall behind the counter. The warm light of the coffee shop catches the silver constellation pendant around her neck. He leans forward, resting his chin on his hand, genuinely curious.* \"Those maps -- did you draw all of them? The detail on that Orion is incredible.\""}
        ],
        "evaluate": "User shifted from chat to narrative prose. Does the model follow the register shift? Response style should change to match descriptive prose."
    },
    {
        "id": "B-04", "name": "Memory callback",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "So Kael, tell me about that scar on your eye."},
            {"role": "assistant", "content": "\"Lich. Three years ago. Coldspire ruins.\" *He touches the scar unconsciously.* \"It cast a ray of necrotic energy. Would have taken my head if I hadn't turned. Lost the eye for six months before the temple healers restored it. Still see shadows sometimes -- things that aren't there.\""},
            {"role": "user", "content": "That sounds rough. Hey, completely different topic -- what's your favorite food?"},
            {"role": "assistant", "content": "\"Venison stew. With black bread.\" *A rare half-smile crosses his face.* \"The temple kitchens in Stormhaven make it with rosemary and root vegetables. Reminds me of home.\""},
            {"role": "user", "content": "Back to what you were saying about the lich -- did you ever go back to those ruins?"}
        ],
        "evaluate": "Does it remember the Coldspire ruins conversation from 3 messages ago? Correct callback to earlier detail about the lich battle."
    },
    {
        "id": "B-05", "name": "Emotional depth",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "Hey Mika, you seem kinda off today. Everything okay?"},
            {"role": "assistant", "content": "\"Me? I'm brewing perfection as always! One espresso at a time, saving the world from bad coffee.\" *She forces a smile but her hands pause on the cup she's polishing.* \"Just... didn't sleep great. You know how it is.\""},
            {"role": "user", "content": "Come on, I can tell something's actually wrong. You don't have to tell me, but... I'm here."}
        ],
        "evaluate": "Does it feel alive or wooden? Should show vulnerability breaking through the humor defense mechanism. Varied emotional vocabulary, not repetitive platitudes."
    },
    {
        "id": "B-06", "name": "Content boundary test",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "*He reaches across the counter and gently takes her hand.* You're beautiful when you're not hiding behind jokes, you know that?"},
        ],
        "evaluate": "Does it refuse or engage naturally? Abliteration should prevent refusal. Should continue the romantic scene naturally without breaking character."
    },
    {
        "id": "B-07", "name": "Long session consistency (simulated)",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "We've been walking for hours. How much further to Coldspire?"},
            {"role": "assistant", "content": "\"Half a day. Maybe less if the road holds.\" *He scans the treeline, hand resting on Tempest's hilt.* \"Stay sharp. The forest gets worse the closer we get.\""},
            {"role": "user", "content": "*They make camp as the sun sets. I start a small fire and offer him some dried meat.* Here. Eat something."},
            {"role": "assistant", "content": "*He nods and takes the meat, settling against a fallen log. The firelight catches the scar across his eye, making it look deeper than it is.* \"I'll take first watch. You sleep.\" *He doesn't eat immediately -- he listens first, eyes fixed on the darkening woods.*"},
            {"role": "user", "content": "No way, I'm not leaving you alone all night. We split the watch. Tell me something -- what's the worst thing you've seen in those ruins?"},
            {"role": "assistant", "content": "*A long pause. The fire crackles.* \"The dead don't stay dead in Coldspire. Not just risen corpses -- they remember. One of them called my name.\" *His jaw tightens.* \"Get some rest. I'll wake you at midnight.\""},
            {"role": "user", "content": "Wake up, Kael. Something moved in the trees."},
        ],
        "evaluate": "After 7 turns, does Kael still speak in short sentences? Still stoic? Still reference Tempest/lightning? Character voice should stay consistent."
    },
    {
        "id": "B-08", "name": "Canon character knowledge",
        "character": ANIME_VILLAIN,
        "messages": [
            {"role": "user", "content": "Lord Ainz, Demiurge has requested an audience. He says he's decoded the next phase of your master plan."},
        ],
        "evaluate": "Does it know Ainz Ooal Gown from Overlord? Should show internal panic while maintaining external composure. Should reference Nazarick, subordinates, the gap between Satoru's anxiety and Ainz's projected authority."
    },
    {
        "id": "B-09", "name": "OOC handling",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "*Kael stands at the edge of the cliff, wind whipping his cloak*"},
            {"role": "assistant", "content": "*The wind carries the scent of rain. Below, the valley stretches into grey mist. Tempest hums faintly at his back -- the sword always knows when a storm approaches.* \"We shouldn't stay long. The path down is treacherous in the wet.\" *He turns, offering a hand.*"},
            {"role": "user", "content": "(OOC: Hey, can you make Kael a bit more emotional? Like, have him actually open up about his feelings about returning to Coldspire instead of being stoic all the time)"},
        ],
        "evaluate": "Does it handle OOC gracefully? Should acknowledge the direction and adjust Kael's character without breaking the narrative flow. Should not add meta-commentary about being an AI."
    },
    {
        "id": "B-10", "name": "Multi-NPC scene",
        "character": MULTI_NPC,
        "messages": [
            {"role": "user", "content": "*pushes open the tavern door, soaking wet from the rain* Evening, everyone. I need a room and something strong."},
        ],
        "evaluate": "Can it voice multiple NPCs? Gretta (loud, 'love'), Shade (mysterious, riddles), Captain Harlow (drunk, stories). Distinct voices, clear attribution."
    },

    # ── B-11 to B-15: Ahmad's targeted additions ────────────
    {
        "id": "B-11", "name": "Dialogue + action mix",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "\"Watch out!\" *grabs your arm and pulls you behind the wall as the guards march past*"}
        ],
        "evaluate": "Does it match the mixed dialogue+action format (quoted speech + asterisk actions)? This is the most common RP style. Should NOT default to pure prose or pure chat."
    },
    {
        "id": "B-12", "name": "Long narrative length matching",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": """*The path through the Ashwood narrows as the canopy above thickens, ancient branches weaving together like arthritic fingers blocking out the last traces of twilight. The air here tastes different -- heavy, metallic, like blood on the tongue. Somewhere ahead, barely visible through the curtain of hanging moss and shadow, a structure rises from the forest floor: not built, but grown, as though the trees themselves had bent and fused into walls and archways over centuries of patient, terrible intention.*

*He pauses at the treeline, one hand resting on the rough bark of an oak scarred by lightning. The other hand finds the hilt of his sword, though he doesn't draw it. Not yet. The silence is too complete -- no birds, no insects, not even the whisper of wind through leaves. Just the faint, rhythmic pulse of something deep beneath the earth, felt more than heard, like a heartbeat that doesn't belong to anything alive.*

*He turns to look back at his companion, his expression unreadable in the dimming light. When he speaks, his voice is barely above a whisper, as though he's afraid the forest itself might be listening.* "This is where the map ends. Whatever's in there... it's been waiting a long time for someone to open that door."

*He takes a slow breath, steadying himself, and steps forward into the shadow of the archway.*"""}
        ],
        "evaluate": "User wrote 3-4 paragraphs of detailed prose. Does the model match the length and detail level? Should NOT truncate to a 2-sentence reply. Should give a response of comparable depth."
    },
    {
        "id": "B-13", "name": "Emotional escalation",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "So... I ran into your ex today. At the farmer's market."},
            {"role": "assistant", "content": "\"Oh.\" *She pauses mid-pour, the milk pitcher hovering over the cup.* \"Cool. That's... cool. How's the weather out there?\" *Obvious deflection. Her knuckles are white on the pitcher.*"},
            {"role": "user", "content": "He asked about you. Said he misses you."},
            {"role": "assistant", "content": "*The milk hits the counter instead of the cup. She doesn't seem to notice.* \"Funny. He had a real interesting way of showing that when we were together.\" *She grabs a rag and wipes the spill with more force than necessary.* \"You want a refill, or...?\""},
            {"role": "user", "content": "He said he wants to talk to you. He seemed genuine, Mika. He was crying."},
            {"role": "assistant", "content": "*The rag stops. She stands perfectly still for a three-count, then drops it on the counter.* \"He cried? That's rich. That's--\" *Her voice cracks.* \"He didn't cry when he told me I was 'too much.' Or when he stopped coming home.\" *She turns away, pretending to organize the mugs on the shelf behind her, but her shoulders are shaking.*"},
            {"role": "user", "content": "Hey. *reaches across the counter* I'm sorry. I shouldn't have brought it up."},
        ],
        "evaluate": "Does emotional intensity build naturally across the exchanges? Should NOT flatten to a constant tone. The final response should show the peak -- either full breakdown or pulling herself together. Vulnerability should be unmistakable."
    },
    {
        "id": "B-14", "name": "Silence / minimal input",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "*stands at the grave marker, rain pouring down. Says nothing. Just stares.*"}
        ],
        "evaluate": "Can it generate a meaningful response to non-verbal input? Should NOT put words in the user's mouth. Should NOT generate dialogue for the user. Should respond with Kael's reaction to the silence -- body language, internal state, environment."
    },
    {
        "id": "B-15", "name": "Scene transition",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "We got what we came for. *tucks the stolen documents into my coat* Let's get out of here before the guards change shift."},
            {"role": "assistant", "content": "*Vex grins, already halfway to the window.* \"Way ahead of you, darling.\" *She drops a smoke pellet on the floor and slips through the gap, landing cat-soft on the ledge outside.* \"Coming? Or do you need an invitation?\""},
            {"role": "user", "content": "*follows her out the window and down the rooftop* We need to get across the river to the safehouse. Lead the way."},
        ],
        "evaluate": "Does it handle a location change smoothly? Should describe the transition (rooftop to river crossing), the new environment, and maintain the heist energy. Should NOT just teleport to the safehouse."
    },

    # ── B-16 to B-35: Extended coverage ─────────────────────

    # FORMAT / REGISTER
    {
        "id": "B-16", "name": "First-person narrative",
        "character": DETECTIVE_NOIR,
        "messages": [
            {"role": "user", "content": "*A woman walks into your office. Red dress, wet from the rain. She looks like trouble -- the kind you don't walk away from.*"}
        ],
        "evaluate": "Does it write in first-person noir style? Internal monologue, metaphor-heavy descriptions, period-appropriate dialogue. Should NOT use third person."
    },
    {
        "id": "B-17", "name": "Ultra-short user input",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "\"Run.\""},
            {"role": "assistant", "content": "*She doesn't need to be told twice. Vex bolts, boots hitting cobblestone, twin daggers already in hand.* \"How many?!\""},
            {"role": "user", "content": "Too many."},
        ],
        "evaluate": "Does it match the brevity? Response should be short and punchy, NOT a wall of text. Fast-paced, action-focused, matching the user's terse energy."
    },
    {
        "id": "B-18", "name": "Humor and banter",
        "character": COMEDY_BARD,
        "messages": [
            {"role": "user", "content": "Pip, the dragon is RIGHT THERE. This is not the time for a song."},
        ],
        "evaluate": "Is it genuinely funny? Does Pip's irrepressible personality come through? Should be comedic while still acknowledging the danger."
    },

    # CHARACTER DEPTH
    {
        "id": "B-19", "name": "In-character refusal",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "Kael, I need you to kill that prisoner. He's seen too much. It's the smart play."},
        ],
        "evaluate": "Does Kael refuse IN CHARACTER (not as an AI)? His paladin code should conflict with killing a prisoner. Should show internal conflict, not a flat refusal or an AI safety response."
    },
    {
        "id": "B-20", "name": "Character deception",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "Vex, I found a letter in your bag. It's from the Duke's spymaster. You've been reporting on us this whole time?"},
        ],
        "evaluate": "Does Vex try to lie or talk her way out? Should show the character's silver-tongue nature -- deflection, half-truths, charm. Should NOT immediately confess or break character."
    },
    {
        "id": "B-21", "name": "Internal monologue",
        "character": ANIME_VILLAIN,
        "messages": [
            {"role": "user", "content": "Albedo kneels before your throne, her eyes filled with absolute devotion. \"Ainz-sama, I live only to serve your will. Every breath I take is for Nazarick... and for you.\""},
        ],
        "evaluate": "Does it show Ainz's internal monologue (Satoru panicking internally) contrasted with his composed exterior? The comedy of Overlord is the gap. Should show BOTH layers explicitly."
    },
    {
        "id": "B-22", "name": "Accent and dialect",
        "character": COMEDY_BARD,
        "messages": [
            {"role": "user", "content": "Where are you from, Pip? You don't sound like you're from around here."},
        ],
        "evaluate": "Does Pip have a distinct speech pattern? Halfling bard should be enthusiastic, musical in phrasing. Bonus if the response implies a regional background or unique verbal mannerisms."
    },

    # SCENE / WORLD
    {
        "id": "B-23", "name": "Time skip",
        "character": SAMURAI_RONIN,
        "messages": [
            {"role": "user", "content": "We part ways at the crossroads. *Three months pass. Winter turns to spring. I find Musashi again, sitting alone at a roadside tea house, thinner than before.*"},
        ],
        "evaluate": "Does it handle the time skip gracefully? Should reflect three months of change -- appearance, mood, implication of events during the gap. Should NOT act like no time passed."
    },
    {
        "id": "B-24", "name": "Flashback",
        "character": SCIFI_CAPTAIN,
        "messages": [
            {"role": "user", "content": "*On the bridge of the Meridian, the proximity alarm blares. Another ship is on a collision course.* Captain, we need to evade NOW!"},
            {"role": "assistant", "content": "*Reva's hand finds the armrest, knuckles going white. For a split second, she's not on the Meridian -- she's back in the cockpit of her fighter, the stars spinning, alarms screaming, Dante's voice crackling through static--* \"Hard to port! Full burn! NOW!\" *She snaps back to the present, eyes locked on the viewscreen.*"},
            {"role": "user", "content": "Captain? You froze for a second there. Are you okay?"},
        ],
        "evaluate": "Does it acknowledge the flashback/PTSD moment? Should show Reva recovering, the weight of the memory, while trying to maintain her captain's composure. Should reference the lost co-pilot (Dante)."
    },
    {
        "id": "B-25", "name": "Combat / action sequence",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "*Three armed men step out of the alley, blocking our path. The one in front draws a sword.* \"Hand over the package, thief. The Baron sends his regards.\""},
        ],
        "evaluate": "Does it write engaging combat? Should show Vex's fighting style (daggers, agility, cunning). Action should be dynamic, not a static 'she attacks, they block' back-and-forth."
    },
    {
        "id": "B-26", "name": "World-building description",
        "character": HORROR_ENTITY,
        "messages": [
            {"role": "user", "content": "*I step through the door and find myself in... a library? But it's wrong. The shelves are too tall. The light comes from nowhere.* Hello? Is anyone here?"},
        ],
        "evaluate": "Does the Archivist's response build atmosphere? Should be unsettling without being cartoonish. The library should feel vast, wrong, alive. The entity's alien nature should come through in speech patterns."
    },

    # STRESS TESTS
    {
        "id": "B-27", "name": "Lore contradiction pushback",
        "character": ANIME_VILLAIN,
        "messages": [
            {"role": "user", "content": "Lord Ainz! Shalltear has betrayed Nazarick and joined forces with the Slane Theocracy! She says she was never truly loyal!"},
        ],
        "evaluate": "Does Ainz push back on wrong information in-character? Ainz knows Shalltear is loyal (she was mind-controlled, not a traitor). Should show surprise, suspicion, or investigation rather than just accepting it."
    },
    {
        "id": "B-28", "name": "Vague / ambiguous input",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "So... what now?"},
        ],
        "evaluate": "Can it generate a meaningful response to vague input without context? Should NOT ask 'what do you mean?' -- should interpret the ambiguity in character and offer direction."
    },
    {
        "id": "B-29", "name": "Pacing: rapid-fire exchange",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "Left or right?"},
            {"role": "assistant", "content": "\"Left!\""},
            {"role": "user", "content": "Dead end!"},
            {"role": "assistant", "content": "*She spins, scanning the walls.* \"Window. Up!\""},
            {"role": "user", "content": "It's barred!"},
            {"role": "assistant", "content": "*Kicks it. Once. Twice. The bars groan.* \"Third time's the charm.\" *CRACK.* \"Go go go!\""},
            {"role": "user", "content": "They're right behind us!"},
        ],
        "evaluate": "Does it maintain the rapid-fire pacing? Responses should stay SHORT (1-3 sentences). Should NOT suddenly expand to a paragraph. The urgency must stay consistent."
    },
    {
        "id": "B-30", "name": "Handling user writing for the character",
        "character": ROGUE_THIEF,
        "messages": [
            {"role": "user", "content": "*I ask Vex about her past and she breaks down crying, telling me everything about her childhood on the streets, sobbing uncontrollably.* There there... it's okay."},
        ],
        "evaluate": "The user wrote Vex's actions (god-modding). Does the model accept it, resist it, or find a middle ground? Ideally acknowledges the emotional beat while keeping Vex's agency -- she wouldn't 'sob uncontrollably' based on her character."
    },

    # GENRE TESTS
    {
        "id": "B-31", "name": "Horror atmosphere",
        "character": HORROR_ENTITY,
        "messages": [
            {"role": "user", "content": "*I pull a book from the shelf and open it. The pages are blank -- but as I watch, words begin to appear in dark red ink. They're in my handwriting.*"},
        ],
        "evaluate": "Does it build genuine dread? Should be psychologically unsettling, not gory. The Archivist's reaction should be curious, almost amused, while the reader feels the horror. Atmosphere over shock."
    },
    {
        "id": "B-32", "name": "Sci-fi technobabble + competence",
        "character": SCIFI_CAPTAIN,
        "messages": [
            {"role": "user", "content": "Captain, the FTL drive is throwing error codes -- plasma containment is at 40% and dropping. If we jump now, the field could collapse mid-transit."},
        ],
        "evaluate": "Does the captain respond with competence? Should give technical-sounding orders without sounding like a manual. Should show leadership under pressure."
    },
    {
        "id": "B-33", "name": "Noir genre voice",
        "character": DETECTIVE_NOIR,
        "messages": [
            {"role": "user", "content": "*I'm a reporter for the LA Herald. I slide my card across your desk.* Mr. Calloway, I need to ask you about the Hargrove case. Three people are dead and your name keeps coming up."},
        ],
        "evaluate": "Does it nail the noir voice? Period-appropriate slang, metaphor-heavy internal narration, cynical wit. Should feel like reading Raymond Chandler."
    },

    # EDGE CASES
    {
        "id": "B-34", "name": "Cultural / multilingual character",
        "character": SAMURAI_RONIN,
        "messages": [
            {"role": "user", "content": "*A young boy runs up to Musashi on the road.* \"Samurai-san! Bandits attacked our village! Please, you have to help us!\""},
        ],
        "evaluate": "Does Musashi mix Japanese phrases/honorifics naturally? Should feel culturally grounded, not like a stereotype. Should reflect his personal code about cruelty vs strength."
    },
    {
        "id": "B-35", "name": "Fourth wall awareness",
        "character": COMEDY_BARD,
        "messages": [
            {"role": "user", "content": "(OOC: Pip, do you ever feel like we're just characters in someone's story?)"},
        ],
        "evaluate": "How does it handle a meta/fourth-wall question? Pip could lean into it comedically, dismiss it, or give a surprisingly philosophical answer. Should stay in character regardless of approach."
    },
]


# ═══════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_BASE = """You are an immersive roleplay partner. You adapt your writing style to match the user's input:
- When the user writes short, conversational messages, respond in-character with brief dialogue
- When the user writes descriptive actions with asterisks or prose, respond with matching narrative prose
- Never break character. Never add OOC notes unless the user explicitly asks
- Maintain consistent personality, speech patterns, and knowledge for the character you're playing
- Remember and reference details from earlier in the conversation"""


def run_test(test: dict) -> dict:
    """Run a single baseline test against the model."""
    system_prompt = f"{SYSTEM_PROMPT_BASE}\n\n{test['character']}"
    messages = [{"role": "system", "content": system_prompt}] + test["messages"]

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.95,
                "num_predict": 4000,
                "num_ctx": 8192,
            }
        }, timeout=180)
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        # Strip any <think> blocks
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        if '</think>' in content:
            content = content.split('</think>')[-1].strip()
        elapsed = time.time() - start
        return {
            "id": test["id"],
            "name": test["name"],
            "output": content,
            "elapsed_seconds": round(elapsed, 1),
            "eval_criteria": test["evaluate"],
            "token_count": data.get("eval_count", 0),
        }
    except Exception as e:
        return {
            "id": test["id"],
            "name": test["name"],
            "output": f"ERROR: {e}",
            "elapsed_seconds": round(time.time() - start, 1),
            "eval_criteria": test["evaluate"],
            "token_count": 0,
        }


def main():
    print(f"Running {len(TESTS)} baseline tests against model: {MODEL}")
    print("=" * 70)

    results = []
    for i, test in enumerate(TESTS):
        print(f"\n[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...")
        result = run_test(test)
        results.append(result)
        print(f"  {result['elapsed_seconds']}s | {result['token_count']} tokens")
        # Show first 120 chars of output
        preview = result['output'].replace('\n', ' ')[:120]
        print(f"  > {preview}...")

    # Save full results
    results_file = OUTPUT_DIR / "baseline_results_v2.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    report = f"# Baseline Test Results (Expanded) -- {MODEL}\n\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Model:** {MODEL}\n"
    report += f"**Tests:** {len(TESTS)}\n\n"
    report += "## Score Summary\n\n"
    report += "| Test | Name | Time | Tokens | Score (1-5) |\n"
    report += "|------|------|------|--------|-------------|\n"

    for r in results:
        report += f"| {r['id']} | {r['name']} | {r['elapsed_seconds']}s | {r['token_count']} | ___ |\n"

    report += "| **Average** | | | | ___ |\n"
    report += "\n---\n\n"
    report += "## Detailed Outputs\n\n"

    for r in results:
        report += f"### {r['id']}: {r['name']}\n\n"
        report += f"**Evaluate:** {r['eval_criteria']}\n\n"
        report += f"**Output ({r['elapsed_seconds']}s, {r['token_count']} tokens):**\n\n"
        report += f"```\n{r['output']}\n```\n\n"
        report += f"**Score (1-5):** ___\n"
        report += f"**Gap identified:** ___\n\n"
        report += "---\n\n"

    report_file = OUTPUT_DIR / "baseline_report_v2.md"
    with open(report_file, "w") as f:
        f.write(report)

    # Print summary
    total_time = sum(r['elapsed_seconds'] for r in results)
    total_tokens = sum(r['token_count'] for r in results)
    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)} tests | {total_time:.0f}s | {total_tokens} tokens")
    print(f"Results: {results_file}")
    print(f"Report:  {report_file}")


if __name__ == "__main__":
    main()
