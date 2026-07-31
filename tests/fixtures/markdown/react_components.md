# Your First Component

*Components* are one of the core concepts of React. They are the foundation upon which you build user interfaces (UI).

## Defining a component

A React component is a JavaScript function that you can sprinkle with markup:

```javascript
export default function Profile() {
  return (
    <img
      src="https://i.imgur.com/MK3eW3Am.jpg"
      alt="Katherine Johnson"
    />
  );
}
```

## Step 1: Export the component

The `export default` prefix is a standard JavaScript syntax. It lets you mark the main function in a file so that you can later import it from other files.

## Step 2: Define the function

With `function Profile() {'{}'}` you define a JavaScript function with the name Profile.

> **Warning:** React components are regular JavaScript functions, but their names must start with a capital letter or they won't work!

## Step 3: Add markup

The component returns an `<img />` tag with `src` and `alt` attributes. This markup looks like HTML, but it is actually JavaScript under the hood!

## Using a component

Now that you've defined your `Profile` component, you can nest it inside other components.

```javascript
function Gallery() {
  return (
    <section>
      <h1>Amazing scientists</h1>
      <Profile />
      <Profile />
      <Profile />
    </section>
  );
}
```
