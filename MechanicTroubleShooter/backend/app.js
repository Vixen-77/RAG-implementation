import express from "express";
import dotenv from "dotenv";

dotenv.config();
import { connectDB } from "./connectDB.js";
import router from "./routers/authRoutes.js";

const app = express();

app.use(express.json())

connectDB();

app.use("/api/auth",router)

try {
  app.listen(3000, () => {
    console.log("Server is running on port 3000");
  });
} catch (error) {
  console.log("Error starting server:", error);
}
