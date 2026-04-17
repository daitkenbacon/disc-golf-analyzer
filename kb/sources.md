# Knowledge Base — Actual Cached Sources

Initial proposed shortlist was based on YouTuber coaches (Overthrow DG,
Danny Lindahl, Disc Golf Strong, Foundation Disc Golf, Paul McBeth). In
practice, most of that content is video-only and not directly scrapable,
and this workspace's `web_fetch` is restricted to Anthropic-owned domains.

What I actually built the KB from: **WebSearch-returned coaching syntheses**
across a broader set of disc golf sites. WebSearch access in this workspace
goes through a separate pipeline and is not subject to the same egress
restrictions, so it returned rich summaries even where `web_fetch` of the
underlying URLs was blocked.

## Sources actually used

Each entry in the KB cites the source URLs it was synthesized from. The
recurring sources include:

- **Ultiworld Disc Golf** — `discgolf.ultiworld.com`
  Tuesday Tips column (Stop Rounding, Don't Fake Your X-Step, Build Your
  Swing, Feet Together, Dissecting the Reach Back Myth, etc.)
- **Inside the Circle DG** — `insidethecircledg.com`
  Coaching blog posts on snap, rounding, plant-foot, breaking down the door
- **Disc Golf Mentor** — `discgolfmentor.com`
  "How to Fix Rounding in 5 Simple Steps"
- **Sweet Az Glass** — `sweetazzglass.com`
  "Master Your X-Step & Plant"
- **DiscSkill** — `discskill.com`
  "Unleash the Backhand Snap", "Run-Up Mastery"
- **HeavyDisc** (Zach Melton) — `heavydisc.com`
  "Improving Back Hand Distance" — on original shortlist
- **UDisc Blog** — `udisc.com/blog`
  Curated "5 Great Videos" posts that reference coaching channels
- **Cherokee Disc Golf** — `cherokeedg.com`
  "Ultimate Guide" to backhand
- **Innova Disc Golf** — `innovadiscs.com/tips`
  "Common Throwing Errors", "How to Throw Backhands Really Far"
- **Dynamic Discs Blog** — `blog.dynamicdiscs.com`
  "Why is footwork so important?"
- **DG Course Review** — `dgcoursereview.com`
  Community threads on specific issues (plant bracing, hip timing, etc.)
- **Pulsea / Avery Jenkins Masterclass** — `pulsea.com`
  Standstill throw fundamentals
- **DiscgolfNOW** — `discgolfnow.com`
  "The 50 Best Disc Golf Drills"
- **DiscGolfReport** — `discgolfreport.com`
  X-step fundamentals

## How this was stored

For each of the six topic KB files (rounding, snap_hit_point, reach_back,
early_shoulders, plant_xstep, drills) the content is a synthesis of the
above sources, with each file's Sources section listing the specific URLs
used for that topic.

## To enable deeper caching later

If full-article transcripts would be valuable, the user would need to
allowlist the above domains (or a subset) under
**Settings → Capabilities → web_fetch allowed hosts** in this workspace.
Once allowlisted, re-running the KB fetch would cache verbatim article
content rather than summaries.

## Not yet covered in the KB (future fetches)

- Deeper biomechanics content from Disc Golf Strong (Seth Burton) — would
  need either his website or YouTube transcripts
- Pro-specific form breakdowns (McBeth, Lizotte, Pierce) — video-heavy,
  harder to cache without YouTube transcript access
- Reddit `/r/discgolf` form-critique conventions — would need reddit.com
  allowlisting
