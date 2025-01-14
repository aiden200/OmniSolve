TEST_DATA = """

```json
{
  "frame_1": {
    "objects": [
      {"name": "train cars", "bbox": [200, 388, 671, 606]},
      {"name": "smoke", "bbox": [130, 389, 339, 502]},
      {"name": "buildings", "bbox": [666, 629, 993, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [114, 647, 300, 732]},
      {"name": "field", "bbox": [0, 0, 227, 197]},
      {"name": "fire trucks", "bbox": [773, 689, 807, 721]},
      {"name": "storage area", "bbox": [632, 8, 824, 190]},
      {"name": "railroad tracks", "bbox": [211, 388, 699, 611]},
      {"name": "other buildings", "bbox": [271, 779, 443, 995]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "field", "relationship": "near", "object_2": "train cars"},
      {"object_1": "fire trucks", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"}
    ],
    "movements": [],
    "summary": "An aerial view of a train derailment. Multiple train cars are derailed and on fire, with smoke billowing from the wreckage. Nearby are buildings, a field, trees, a storage area, and fire trucks responding to the incident."
  },
  "frame_2": {
    "objects": [
      {"name": "train cars", "bbox": [197, 388, 675, 604]},
      {"name": "smoke", "bbox": [128, 384, 339, 503]},
      {"name": "buildings", "bbox": [666, 626, 999, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [114, 647, 302, 733]},
      {"name": "field", "bbox": [0, 0, 226, 195]},
      {"name": "fire trucks", "bbox": [775, 689, 809, 721]},
      {"name": "storage area", "bbox": [633, 8, 826, 190]},
      {"name": "railroad tracks", "bbox": [209, 385, 701, 613]},
      {"name": "other buildings", "bbox": [273, 779, 444, 995]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "field", "relationship": "near", "object_2": "train cars"},
      {"object_1": "fire trucks", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"}
    ],
    "movements": [],
    "summary": "The same scene as the previous frame, showing a train derailment with fire and smoke.  There are no apparent changes in object positions or movement."
  },
  "frame_3": {
    "objects": [
      {"name": "train cars", "bbox": [201, 384, 677, 606]},
      {"name": "smoke", "bbox": [128, 385, 342, 500]},
      {"name": "buildings", "bbox": [668, 628, 999, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [114, 647, 301, 731]},
      {"name": "field", "bbox": [0, 0, 225, 197]},
      {"name": "fire trucks", "bbox": [775, 688, 810, 721]},
      {"name": "storage area", "bbox": [633, 7, 825, 189]},
      {"name": "railroad tracks", "bbox": [209, 386, 701, 612]},
      {"name": "other buildings", "bbox": [273, 780, 444, 995]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "field", "relationship": "near", "object_2": "train cars"},
      {"object_1": "fire trucks", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"}
    ],
    "movements": [],
    "summary": "The scene remains unchanged from the previous frame.  A train derailment with fire and smoke is visible, with surrounding buildings, a field, and emergency response vehicles."
  },
  "frame_4": {
    "objects": [
      {"name": "train cars", "bbox": [214, 388, 671, 605]},
      {"name": "smoke", "bbox": [132, 391, 338, 499]},
      {"name": "buildings", "bbox": [667, 626, 999, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [115, 648, 301, 732]},
      {"name": "field", "bbox": [0, 0, 226, 196]},
      {"name": "fire trucks", "bbox": [777, 690, 809, 720]},
      {"name": "storage area", "bbox": [633, 8, 825, 189]},
      {"name": "railroad tracks", "bbox": [211, 385, 699, 612]},
      {"name": "other buildings", "bbox": [273, 779, 444, 995]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "field", "relationship": "near", "object_2": "train cars"},
      {"object_1": "fire trucks", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"}
    ],
    "movements": [],
    "summary": "The scene remains unchanged.  A train derailment with fire and smoke is visible near buildings, a field, and a storage area. Emergency response vehicles are present."
  },
  "frame_5": {
    "objects": [
      {"name": "train cars", "bbox": [276, 358, 729, 609]},
      {"name": "smoke", "bbox": [285, 400, 455, 508]},
      {"name": "buildings", "bbox": [736, 632, 999, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [214, 302, 420, 396]},
      {"name": "storage area", "bbox": [622, 13, 805, 198]},
      {"name": "railroad tracks", "bbox": [296, 363, 727, 620]},
      {"name": "emergency response vehicles", "bbox": [793, 691, 835, 752]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"},
      {"object_1": "emergency response vehicles", "relationship": "near", "object_2": "train cars"}
    ],
    "movements": [
      {"object": "camera", "trajectory": "moved closer to the train derailment"}
    ],
    "summary": "The camera zooms in on the derailed train cars, showing a clearer view of the damage and smoke.  The focus is now primarily on the accident scene, with fewer surrounding elements visible."
  },
  "frame_6": {
    "objects": [
      {"name": "train cars", "bbox": [307, 358, 724, 600]},
      {"name": "smoke", "bbox": [307, 400, 465, 507]},
      {"name": "buildings", "bbox": [734, 630, 999, 999]},
      {"name": "ground", "bbox": [0, 0, 999, 999]},
      {"name": "trees", "bbox": [216, 302, 418, 396]},
      {"name": "storage area", "bbox": [621, 13, 806, 198]},
      {"name": "railroad tracks", "bbox": [302, 362, 726, 619]},
      {"name": "emergency response vehicles", "bbox": [791, 691, 835, 751]}
    ],
    "relationships": [
      {"object_1": "train cars", "relationship": "on", "object_2": "ground"},
      {"object_1": "smoke", "relationship": "from", "object_2": "train cars"},
      {"object_1": "buildings", "relationship": "near", "object_2": "train cars"},
      {"object_1": "trees", "relationship": "near", "object_2": "train cars"},
      {"object_1": "storage area", "relationship": "near", "object_2": "train cars"},
      {"object_1": "train cars", "relationship": "on", "object_2": "railroad tracks"},
      {"object_1": "emergency response vehicles", "relationship": "near", "object_2": "train cars"}
    ],
    "movements": [],
    "summary": "Similar to the previous frame, the camera is zoomed in on the train derailment, and the view is primarily focused on the damaged train cars and smoke. There are no apparent changes in object positions or movement."
  },
  "video_summary": "The video shows an aerial view of a train derailment and subsequent fire. The video begins with a wider shot showing the overall scene, then zooms in closer to the accident.  The key objects are the derailed and burning train cars, smoke, surrounding buildings, a field, trees, storage areas, and emergency response vehicles. The video mainly shows a static scene; there are no significant movements of objects besides the initial camera zoom."
}
```
"""