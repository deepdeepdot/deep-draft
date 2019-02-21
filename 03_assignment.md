---
1. Transform expression using parenthesis
Support unary -

---

2. Print indented tree

In the class Node, if you replace 'print_...' with __str__,
we can use the print() statement like the following

print(root) 

What's the DOM?


### HTML

html = ["html", [
    "head", [
        "title", "Learning the Buddha"
    ],
    "body", [
        ["h1", "The Four Noble Truths")]
        ["ul", [
            ["li", "What's suffering?"],
            ["li", "The cause of suffering"],
            ["li", "The end of suffering"],
            ["li", "A path to the end of suffering"],
        ])
    ])
])

1. Write a recursive function to print this hmtl tree
    Hint: from the looks, if the second item is an array, we need to recurse

2. Add indentation when printing the html tree

Implement the dom class, so that we can support this design

    dom(tag, id, tagAttributes, text, children)

html = dom("html", children=[
    dom("head", children=[
        dom("title", text="deep learning for dummies")
    ]),
    dom("body", children=[
        dom("h1", text="The four noble truths"),
        dom("ul", children=[
            dom("li", text="What's suffering?"),
            dom("li", text="The cause of suffering"),
            dom("li", text="The end of suffering"),
            dom("li", text="A path to the end of suffering"),
        ]),
        dom("p", id="last_mark", text="")
    ])
])

Implement a recursive indented printing function


How about supporting the image tag?

img = dom("img", {
    source: "https://minecraft.net/favicon-32x32.png",
    title: "Minecraft"
});


html.find("last_mark").appendChild(img)

You need to implement the recursive find() and the appendChild(node) methods.

And print the html recursively


#### Tensorflow practice

Compute and display the Bell Curve using tensorflow instead of the numpy library
