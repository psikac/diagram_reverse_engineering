// const image_input = document.querySelector("#image_input");
window.onload = function () {
  const image_input = document.querySelector("#image_input");

  image_input.addEventListener("change", function () {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const uploaded_image = reader.result.replace("data:", "")
      .replace(/^.+,/, "");
      const json = JSON.stringify({
        image: uploaded_image
      })
      const url = 'http://localhost:5000/image';
      const data = { title: "The Matrix", year: "1994" };

      fetch(
        url,
        {
          headers: { "Access-Control-Allow-Origin": "*" },
          headers: { "Content-Type": "application/json" },
          body: json,
          method: "POST"
        }
      )
        .then(data => data.json())
        .then((json) => {
          console.log(json);
          returnedObject = JSON.parse(json);
          console.log(returnedObject)
        });

      document.querySelector("#display_image").style.backgroundImage = `url(${uploaded_image})`;
    });
    reader.readAsDataURL(this.files[0]);
  });

}


