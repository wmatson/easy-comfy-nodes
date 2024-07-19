# easy-comfy-nodes

A handful of utility nodes for ComfyUI

## Development

Clone this repo to your ComfyUI installation under `ComfyUI/custom_nodes/easy-comfy-nodes`, restart ComfyUI whenever you make a change
`__init__.py` is the entrypoint ComfyUI uses to discover the custom nodes, see [example_node in the comfy repo](https://github.com/comfyanonymous/ComfyUI/blob/eb5c991a8c24873b3efba747ec4466d4f2c986db/custom_nodes/example_node.py.example) for details

Prefix all easy-comfy-nodes with `EZ`

## Nodes

### Animation-related Nodes
- VideoCombine (removed): use S3 Upload with [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)'s `Video Combine` instead
- S3 Upload: takes `filenames` as from the "Video Combine" node in [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), uploads them to the relevant s3 bucket and object name using the default credential provider

### HTTP Nodes
Nodes for handling http requests as part of a workflow, these can be used to create webhooks and the like for different parts of a workflow

- HTTP POST: emits a POST request to `url` with dict `body` as JSON
- Load Img From URL: loads an image from a url, this also works for data urls and s3 uris (using the default credential provider)
- Load Img Batch From URLs: loads a batch of images from a set of URLs on separate lines, loaded just as the single-url node

### Dict Nodes
Some nodes for handling dictionary/map data structures

- Empty Dict: returns an empty Dict
- Assoc X: Associates `key` with `value` of type X